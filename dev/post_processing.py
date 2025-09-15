import numpy as np
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

import pretty_midi

LABELS_5 = [35, 38, 47, 42, 49]
FRAME_RNN_THRESHOLDS = [0.22, 0.24, 0.32, 0.22, 0.30]


class NotePeakPickingProcessor:
    """
    Lightweight reimplementation of madmom.features.notes.NotePeakPickingProcessor used in ADTOF.
    Operates on a 1D activation array sampled at a given fps and returns onset times (in seconds).
    
    Parameters mirror the original defaults used in ADTOF's PeakPicking.setParameters:
    - threshold: detection threshold on activation
    - pre_avg, post_avg: moving average window sizes (seconds) for local mean subtraction (set to zero to disable)
    - pre_max, post_max: local maximum window sizes (seconds) for peak localization
    - combine: minimum time (seconds) between successive peaks; close peaks are merged keeping the highest one
    - fps: frames per second of the activation sequence
    
    For simplicity and reproducibility, smoothing and local average are optional; core logic is threshold + local maxima + non-maximum suppression + combine.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        pre_avg: float = 0.1,
        post_avg: float = 0.01,
        pre_max: float = 0.02,
        post_max: float = 0.01,
        combine: float = 0.02,
        fps: int = 100,
    ):
        self.threshold = float(threshold)
        self.pre_avg = float(pre_avg)
        self.post_avg = float(post_avg)
        self.pre_max = float(pre_max)
        self.post_max = float(post_max)
        self.combine = float(combine)
        self.fps = int(fps)

    def _moving_average(self, x: np.ndarray, win_left: int, win_right: int) -> np.ndarray:
        if win_left <= 0 and win_right <= 0:
            return x
        # Causal-left + right window mean centered at each frame
        kernel = np.ones(win_left + 1 + win_right, dtype=np.float32)
        kernel /= kernel.size
        padded = np.pad(x, (win_left, win_right), mode='edge')
        ma = np.convolve(padded, kernel, mode='valid')
        return ma.astype(np.float32, copy=False)

    def _local_maxima(self, x: np.ndarray, w_left: int, w_right: int) -> np.ndarray:
        # A frame i is a local maxima if it equals the maximum within [i-w_left, i+w_right]
        if w_left <= 0 and w_right <= 0:
            return x
        size = w_left + 1 + w_right
        # Pad to handle borders similar to madmom (edge padding)
        padded = np.pad(x, (w_left, w_right), mode='edge')
        # Build a stacked view of all window-aligned slices and take max across the window axis
        # windows[k] corresponds to x shifted by k-w_left
        windows = np.stack([padded[k:k + len(x)] for k in range(0, size)], axis=0)
        window_max = windows.max(axis=0)
        return window_max

    def process(self, activation: np.ndarray) -> List[Tuple[float, int]]:
        """
        Return list of (time, pitch) tuples; pitch is unused here but kept for API parity with original usage.
        """
        act = np.asarray(activation, dtype=np.float32).reshape(-1)
        # Optional mean subtraction using moving average windows
        pre_avg_frames = int(round(self.pre_avg * self.fps))
        post_avg_frames = int(round(self.post_avg * self.fps))
        if pre_avg_frames > 0 or post_avg_frames > 0:
            avg = self._moving_average(act, pre_avg_frames, post_avg_frames)
            proc = np.maximum(0.0, act - avg)
        else:
            proc = act

        # Local maxima support
        pre_max_frames = int(round(self.pre_max * self.fps))
        post_max_frames = int(round(self.post_max * self.fps))
        if pre_max_frames > 0 or post_max_frames > 0:
            window_max = self._local_maxima(proc, pre_max_frames, post_max_frames)
            is_peak = (proc >= window_max) & (proc >= self.threshold)
        else:
            is_peak = proc >= self.threshold

        peak_indices = np.where(is_peak)[0]
        if peak_indices.size == 0:
            return []

        # Combine peaks that are closer than `combine`
        combine_frames = max(1, int(round(self.combine * self.fps)))
        kept: List[int] = []
        current_group: List[int] = [int(peak_indices[0])]
        for idx in peak_indices[1:]:
            if idx - current_group[-1] <= combine_frames:
                current_group.append(int(idx))
            else:
                # keep the highest activation within the group
                best = int(max(current_group, key=lambda i: proc[i]))
                kept.append(best)
                current_group = [int(idx)]
        # finalize last group
        best = int(max(current_group, key=lambda i: proc[i]))
        kept.append(best)

        # Convert to seconds
        times = [i / float(self.fps) for i in kept]
        return [(t, 0) for t in times]


class PeakPicker:
    """
    Peak picking over multi-class frame activations with per-class thresholds.
    Mimics adtof.model.peakPicking.PeakPicking.predict behavior and API.
    """
    def __init__(self, thresholds: Sequence[float] | float = FRAME_RNN_THRESHOLDS, fps: int = 100):
        if isinstance(thresholds, (list, tuple, np.ndarray)):
            self.processors = [
                NotePeakPickingProcessor(threshold=float(t), pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=fps)
                for t in thresholds
            ]
        else:
            self.processors = NotePeakPickingProcessor(threshold=float(thresholds), pre_avg=0.1, post_avg=0.01, pre_max=0.02, post_max=0.01, combine=0.02, fps=fps)
        self.fps = fps

    def pick(self, activations: np.ndarray, labels: Sequence[int] = LABELS_5, label_offset: int = 0) -> List[Dict[int, List[float]]]:
        """
        Args:
            activations: [batch, time, classes] or [time, classes]
            labels: MIDI pitches per class index
            label_offset: offset in frames to add to onset times
        Returns:
            List of dict {label_midi: [times_in_sec]}
        """
        x = np.asarray(activations, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        B, T, C = x.shape
        assert C == len(labels), f"Expected {len(labels)} classes, got {C}"

        process_list: List[NotePeakPickingProcessor]
        if isinstance(self.processors, list) and len(self.processors) == len(labels):
            process_list = self.processors
        else:
            process_list = [self.processors for _ in range(len(labels))]  # type: ignore[list-item]

        time_offset = label_offset / float(self.fps)
        results: List[Dict[int, List[float]]] = []
        for b in range(B):
            result: Dict[int, List[float]] = {}
            for i, lab in enumerate(labels):
                proc = process_list[i]
                peaks = proc.process(x[b, :, i])
                times = [t + time_offset for (t, _unused_pitch) in peaks]
                result[int(lab)] = times
            results.append(result)
        return results


def activations_to_pretty_midi(
    peaks: Dict[int, List[float]],
    velocity: int = 100,
    note_duration: float = 0.1,
    program: int = 1,
    is_drum: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI object with a single drum instrument from picked peaks.
    - peaks: dict {midi_pitch: [onset_times_seconds]}
    - velocity: MIDI velocity 0-127 for all notes
    - note_duration: fixed duration in seconds per note
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
    for pitch, times in peaks.items():
        for t in times:
            note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=float(t), end=float(t + note_duration))
            instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi
