import os
import argparse
import torch
from adtof_pytorch import create_frame_rnn_model, load_audio_for_model, calculate_n_bins
from post_processing import PeakPicker, FRAME_RNN_THRESHOLDS, LABELS_5, activations_to_pretty_midi
from pathlib import Path
from typing import Sequence


# -----------------------------
# CLI / Test script (__main__)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADTOF PyTorch model, peak-pick, and export MIDI")
    parser.add_argument('--audio', type=str, default='test.wav', help='Path to input audio file')
    parser.add_argument('--out', type=str, default='test.mid', help='Output MIDI path')
    parser.add_argument('--threshold', type=float, default=None, help='Uniform threshold for all classes (overrides per-class if set)')
    parser.add_argument('--thresholds', type=str, default='', help='Comma-separated per-class thresholds (e.g., 0.22,0.24,0.32,0.22,0.30)')
    parser.add_argument('--fps', type=int, default=100, help='Frames per second of model activations')
    parser.add_argument('--weights', type=str, default='adtof_frame_rnn_pytorch_weights.pth', help='Optional path to PyTorch model weights (.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    args = parser.parse_args()

    assert os.path.exists(args.audio), f"Audio file not found: {args.audio}"
    assert args.device in ['cuda', 'cpu'], f"Invalid device: {args.device}"

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, using CPU")
            device = 'cpu'
        else:
            device = 'cuda'
    else:
        device = 'cpu'

    # Prepare model
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    model.eval()

    if args.weights and os.path.exists(args.weights):
        from adtof_pytorch import load_pytorch_weights
        model = load_pytorch_weights(model, args.weights, strict=False)

    model.to(device)

    # Load audio and run model
    x = load_audio_for_model(args.audio)
    x = x.to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # [1, time, classes]

    # Resolve thresholds
    thresholds_to_use: Sequence[float] | float
    if args.thresholds:
        try:
            thresholds_to_use = [float(v.strip()) for v in args.thresholds.split(',') if v.strip() != '']
        except Exception as e:
            raise ValueError(f"Failed to parse --thresholds '{args.thresholds}': {e}")
    elif args.threshold is not None:
        thresholds_to_use = float(args.threshold)
    else:
        thresholds_to_use = FRAME_RNN_THRESHOLDS

    # Peak picking
    print(f"pred shape: {pred.shape}")
    picker = PeakPicker(thresholds=thresholds_to_use, fps=args.fps)
    picked_list = picker.pick(pred, labels=LABELS_5, label_offset=0)
    peaks_dict = picked_list[0]
    print(f"peaks_dict: {peaks_dict}")

    # Export MIDI
    midi = activations_to_pretty_midi(peaks_dict, velocity=100, note_duration=0.1, program=1, is_drum=True)

    # Handle output path
    out_path = Path(args.out)
    if out_path.exists():
        print(f"Warning: Overwriting existing file {args.out}")
        os.remove(args.out)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = str(out_path)

    midi.write(out_path)
    print(f"Wrote MIDI to {out_path}")