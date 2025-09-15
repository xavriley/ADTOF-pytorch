from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Union

import torch

from .model import (
    ADTOFFrameRNN,
    create_frame_rnn_model,
    calculate_n_bins,
    load_audio_for_model,
    load_pytorch_weights,
)
from .post_processing import (
    PeakPicker,
    FRAME_RNN_THRESHOLDS,
    LABELS_5,
    activations_to_pretty_midi,
)


def get_default_weights_path() -> Optional[str]:
    """Return the installed default weights path if available."""
    try:
        from importlib.resources import files

        return str(files(__package__) / "data" / "adtof_frame_rnn_pytorch_weights.pth")
    except Exception:
        return None


def transcribe_to_midi(
    audio: Union[str, Path],
    midi_out: Union[str, Path],
    *,
    threshold: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
    fps: int = 100,
    weights: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> Path:
    """
    Transcribe an audio file to a MIDI file using the ADTOF Frame_RNN model.

    Args:
        audio: Path to input audio file
        midi_out: Path to write the output MIDI file
        threshold: Uniform threshold for all classes (overrides per-class thresholds)
        thresholds: Per-class thresholds in order of LABELS_5
        fps: Frames per second of model activations
        weights: Path to model weights (.pth). Defaults to packaged weights if available
        device: 'cuda' or 'cpu'

    Returns:
        Path to the written MIDI file
    """
    audio_path = Path(audio)
    out_path = Path(midi_out)
    assert audio_path.exists(), f"Audio file not found: {audio}"
    assert device in ("cuda", "cpu"), f"Invalid device: {device}"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Prepare model
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    model.eval()

    # Resolve weights
    weights_path: Optional[Path]
    if weights is not None:
        weights_path = Path(weights)
    else:
        default_w = get_default_weights_path()
        weights_path = Path(default_w) if default_w else None

    if weights_path is not None and weights_path.exists():
        model = load_pytorch_weights(model, str(weights_path), strict=False)

    model.to(device)

    # Load audio and run model
    x = load_audio_for_model(str(audio_path))
    x = x.to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # [1, time, classes]

    # Threshold resolution
    if thresholds is not None and len(list(thresholds)) > 0:
        thresholds_to_use: Sequence[float] | float = [float(v) for v in thresholds]
    elif threshold is not None:
        thresholds_to_use = float(threshold)
    else:
        thresholds_to_use = FRAME_RNN_THRESHOLDS

    # Peak picking
    picker = PeakPicker(thresholds=thresholds_to_use, fps=fps)
    picked_list = picker.pick(pred, labels=LABELS_5, label_offset=0)
    peaks_dict = picked_list[0]

    # Export MIDI
    midi = activations_to_pretty_midi(peaks_dict, velocity=100, note_duration=0.1, program=1, is_drum=True)

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))
    return out_path


__all__ = [
    "ADTOFFrameRNN",
    "create_frame_rnn_model",
    "calculate_n_bins",
    "load_audio_for_model",
    "load_pytorch_weights",
    "PeakPicker",
    "FRAME_RNN_THRESHOLDS",
    "LABELS_5",
    "activations_to_pretty_midi",
    "get_default_weights_path",
    "transcribe_to_midi",
]
