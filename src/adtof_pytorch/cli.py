import argparse
import os
from pathlib import Path
from typing import Sequence

from . import transcribe_to_midi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADTOF PyTorch model, peak-pick, and export MIDI")
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--out', type=str, required=True, help='Output MIDI path')
    parser.add_argument('--threshold', type=float, default=None, help='Uniform threshold for all classes (overrides per-class if set)')
    parser.add_argument('--thresholds', type=str, default='', help='Comma-separated per-class thresholds (e.g., 0.22,0.24,0.32,0.22,0.30)')
    parser.add_argument('--fps', type=int, default=100, help='Frames per second of model activations')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to PyTorch model weights (.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert os.path.exists(args.audio), f"Audio file not found: {args.audio}"
    assert args.device in ['cuda', 'cpu'], f"Invalid device: {args.device}"

    thresholds_to_use: Sequence[float] | None = None
    if args.thresholds:
        try:
            thresholds_to_use = [float(v.strip()) for v in args.thresholds.split(',') if v.strip() != '']
        except Exception as e:
            raise ValueError(f"Failed to parse --thresholds '{args.thresholds}': {e}")

    transcribe_to_midi(
        audio=args.audio,
        midi_out=args.out,
        threshold=args.threshold,
        thresholds=thresholds_to_use,
        fps=args.fps,
        weights=args.weights,
        device=args.device,
    )
    print(f"Wrote MIDI to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
