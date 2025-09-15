import argparse
import os
from pathlib import Path
from typing import Sequence, List

import torch

from . import (
    calculate_n_bins,
    create_frame_rnn_model,
    load_audio_for_model,
    load_pytorch_weights,
    PeakPicker,
    FRAME_RNN_THRESHOLDS,
    LABELS_5,
    activations_to_pretty_midi,
    get_default_weights_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADTOF PyTorch model, peak-pick, and export MIDI")
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file or directory containing audio files')
    parser.add_argument('--out', type=str, required=False, default=None, help='Output MIDI path (file for single input, or directory for batch)')
    parser.add_argument('--out-suffix', type=str, default='_adtof-pt.mid', help='Suffix to append to input stem for output filename when --out is not a file path')
    parser.add_argument('--threshold', type=float, default=None, help='Uniform threshold for all classes (overrides per-class if set)')
    parser.add_argument('--thresholds', type=str, default='', help='Comma-separated per-class thresholds (e.g., 0.22,0.24,0.32,0.22,0.30)')
    parser.add_argument('--fps', type=int, default=100, help='Frames per second of model activations')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to PyTorch model weights (.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert os.path.exists(args.audio), f"Audio path not found: {args.audio}"
    assert args.device in ['cuda', 'cpu'], f"Invalid device: {args.device}"

    thresholds_to_use: Sequence[float] | None = None
    if args.thresholds:
        try:
            thresholds_to_use = [float(v.strip()) for v in args.thresholds.split(',') if v.strip() != '']
        except Exception as e:
            raise ValueError(f"Failed to parse --thresholds '{args.thresholds}': {e}")

    audio_path = Path(args.audio)

    # Build list of input audio files
    input_files: List[Path]
    exts = {'.wav', '.mp3', '.flac'}
    if audio_path.is_file():
        input_files = [audio_path]
    elif audio_path.is_dir():
        input_files = sorted([p for p in audio_path.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not input_files:
            raise FileNotFoundError(f"No audio files with extensions {sorted(exts)} found in directory: {audio_path}")
    else:
        raise FileNotFoundError(f"Invalid --audio path: {audio_path}")

    # Resolve device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Prepare model once
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    model.eval()

    # Resolve weights
    weights_path: Path | None
    if args.weights is not None:
        weights_path = Path(args.weights)
    else:
        default_w = get_default_weights_path()
        weights_path = Path(default_w) if default_w else None

    if weights_path is not None and weights_path.exists():
        model = load_pytorch_weights(model, str(weights_path), strict=False)

    model.to(device)

    # Resolve threshold configuration
    if thresholds_to_use is not None and len(list(thresholds_to_use)) > 0:
        resolved_thresholds: Sequence[float] | float = [float(v) for v in thresholds_to_use]
    elif args.threshold is not None:
        resolved_thresholds = float(args.threshold)
    else:
        resolved_thresholds = FRAME_RNN_THRESHOLDS

    picker = PeakPicker(thresholds=resolved_thresholds, fps=args.fps)

    # Determine output handling
    out_arg = Path(args.out) if args.out is not None else None
    treat_out_as_dir = False
    if out_arg is not None:
        if len(input_files) > 1:
            # For multiple inputs, --out must be a directory
            treat_out_as_dir = True
        else:
            # Single input: treat as file unless it's a directory
            treat_out_as_dir = out_arg.exists() and out_arg.is_dir()

    for in_file in input_files:
        # Compute output path per file
        if out_arg is None:
            out_path = in_file.with_name(in_file.stem + args.out_suffix)
        else:
            if treat_out_as_dir:
                out_arg.mkdir(parents=True, exist_ok=True)
                out_path = out_arg / (in_file.stem + args.out_suffix)
            else:
                out_path = out_arg

        # Load audio and run inference
        x = load_audio_for_model(str(in_file))
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()

        # Peak picking and MIDI export
        picked_list = picker.pick(pred, labels=LABELS_5, label_offset=0)
        peaks_dict = picked_list[0]
        midi = activations_to_pretty_midi(peaks_dict, velocity=100, note_duration=0.1, program=1, is_drum=True)

        if out_path.exists():
            out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(out_path))
        print(f"Wrote MIDI to {out_path.resolve()}")


if __name__ == "__main__":
    main()
