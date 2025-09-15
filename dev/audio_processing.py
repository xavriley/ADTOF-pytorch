"""
Clean audio processing implementation for ADTOF without madmom dependency.

This module reimplements the necessary audio processing pipeline:
1. Audio loading and resampling
2. Framing with overlap
3. Short-Time Fourier Transform (STFT)
4. Logarithmic filterbank for frequency analysis

Replaces madmom functionality with librosa and numpy implementations.
"""

import numpy as np
import librosa
from typing import Tuple


class AudioProcessor:
    """
    Clean audio processor that replaces madmom functionality.
    
    Implements the same processing pipeline as the original ADTOF:
    - Load audio at specified sample rate
    - Frame the signal with overlap
    - Compute STFT
    - Apply logarithmic filterbank
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        fps: int = 100,  # Frames per second for output
        frame_size: int = 2048,
        bands_per_octave: int = 12,
        fmin: float = 20.0,
        fmax: float = 20000.0,
        n_channels: int = 1,
        normalize: bool = False
    ):
        self.sample_rate = sample_rate
        self.fps = fps
        self.frame_size = frame_size
        self.bands_per_octave = bands_per_octave
        self.fmin = fmin
        self.fmax = fmax
        self.n_channels = n_channels
        self.normalize = normalize
        
        # Calculate derived parameters
        self.hop_length = int(np.round(sample_rate / fps))
        self.n_fft = frame_size
        
        # Pre-calculate filterbank
        self._setup_filterbank()
    
    def _setup_filterbank(self):
        """
        Setup logarithmic filterbank matching madmom's LogarithmicFilterbank.
        """
        # Use madmom's exact frequency calculation
        target_frequencies = self._log_frequencies(self.bands_per_octave, self.fmin, self.fmax)
        
        # FFT frequency bins (matching madmom's fft_frequencies; exclude Nyquist)
        fft_freqs = np.fft.fftfreq(self.n_fft, 1 / self.sample_rate)[: self.n_fft // 2]
        
        # Convert frequencies to bin indices (matching madmom's frequencies2bins)
        bins = self._frequencies_to_bins(target_frequencies, fft_freqs, unique_bins=True)
        
        # Create triangular filters (matching madmom's TriangularFilter.filters)
        self.filterbank = self._create_madmom_filterbank(bins, len(fft_freqs)).astype(np.float32, copy=False)
        self.n_bins = self.filterbank.shape[0]
    
    def _log_frequencies(self, bands_per_octave: int, fmin: float, fmax: float) -> np.ndarray:
        """
        Calculate base-2 logarithmic frequencies like madmom.audio.filters.log_frequencies.
        f_k = fmin * 2**(k / bands_per_octave), up to fmax.
        """
        freqs = []
        factor = 2.0 ** (1.0 / bands_per_octave)
        f = fmin
        # Include both endpoints similarly to madmom
        while f <= fmax * (1.0 + 1e-12):
            freqs.append(f)
            f *= factor
        return np.array(freqs, dtype=float)
    
    def _frequencies_to_bins(self, frequencies: np.ndarray, fft_freqs: np.ndarray, unique_bins: bool = True) -> np.ndarray:
        """
        Map target frequencies to nearest FFT bin indices (madmom uses nearest bins).
        """
        # Vectorized nearest-bin mapping
        # For each target frequency, find index of closest fft frequency
        # Compute absolute differences matrix in a memory-efficient way
        bins = np.empty(len(frequencies), dtype=int)
        for i, f in enumerate(frequencies):
            bins[i] = int(np.argmin(np.abs(fft_freqs - f)))
        
        if unique_bins:
            # Remove duplicates while preserving order and ensuring strictly increasing indices
            unique_bins_array = []
            last_bin = -1
            for bin_idx in bins:
                if bin_idx > last_bin:
                    unique_bins_array.append(int(bin_idx))
                    last_bin = int(bin_idx)
            bins = np.array(unique_bins_array, dtype=int)
        
        return bins
    
    def _create_madmom_filterbank(self, bins: np.ndarray, n_fft_bins: int) -> np.ndarray:
        """
        Create triangular filterbank matching madmom's TriangularFilter.filters.
        """
        n_filters = len(bins) - 2  # Need at least 3 points for triangular filter
        filterbank = np.zeros((n_filters, n_fft_bins), dtype=np.float32)
        
        for i in range(n_filters):
            # Three consecutive bins for triangular filter
            left_bin = int(bins[i])
            center_bin = int(bins[i + 1])
            right_bin = int(bins[i + 2])

            # Handle degenerate too-small filters like madmom
            if right_bin - left_bin < 2:
                if 0 <= left_bin < n_fft_bins:
                    filterbank[i, left_bin] = 1.0
                continue

            # Left slope: values from 0 up to just below 1 (exclude center)
            if center_bin > left_bin:
                for b in range(left_bin, center_bin):
                    filterbank[i, b] = (b - left_bin) / float(center_bin - left_bin)

            # Peak at center
            if 0 <= center_bin < n_fft_bins:
                filterbank[i, center_bin] = 1.0

            # Right slope: strictly after center, exclude right endpoint
            if right_bin > center_bin + 0:
                for b in range(center_bin + 1, min(right_bin, n_fft_bins)):
                    filterbank[i, b] = (right_bin - b) / float(right_bin - center_bin)
        
        # Normalize filters (norm=True in madmom)
        filter_sums = np.sum(filterbank, axis=1, keepdims=True)
        filter_sums[filter_sums == 0] = 1
        filterbank = filterbank / filter_sums
        
        return filterbank

    def _create_triangular_filterbank(self, freq_centers: np.ndarray, fft_freqs: np.ndarray) -> np.ndarray:
        """
        Create triangular filterbank similar to madmom's TriangularFilter.
        
        Args:
            freq_centers: Center frequencies for filters
            fft_freqs: FFT frequency bins
            
        Returns:
            Filterbank matrix of shape [n_filters, n_fft_bins]
        """
        n_filters = len(freq_centers) - 2  # Exclude first and last for triangular shape
        n_fft_bins = len(fft_freqs)
        
        filterbank = np.zeros((n_filters, n_fft_bins))
        
        for i in range(n_filters):
            # Triangular filter with three points
            left_freq = freq_centers[i]
            center_freq = freq_centers[i + 1] 
            right_freq = freq_centers[i + 2]
            
            # Find corresponding FFT bins
            left_bin = np.argmin(np.abs(fft_freqs - left_freq))
            center_bin = np.argmin(np.abs(fft_freqs - center_freq))
            right_bin = np.argmin(np.abs(fft_freqs - right_freq))
            
            # Create triangular filter
            # Left slope: 0 to 1
            if center_bin > left_bin:
                filterbank[i, left_bin:center_bin+1] = np.linspace(0, 1, center_bin - left_bin + 1)
            
            # Right slope: 1 to 0  
            if right_bin > center_bin:
                filterbank[i, center_bin:right_bin+1] = np.linspace(1, 0, right_bin - center_bin + 1)
        
        # Normalize filters (similar to madmom's norm_filters=True)
        filter_sums = np.sum(filterbank, axis=1, keepdims=True)
        filter_sums[filter_sums == 0] = 1  # Avoid division by zero
        filterbank = filterbank / filter_sums
        
        return filterbank
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file with specified parameters.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio signal of shape [n_samples] or [n_samples, n_channels]
        """
        # Load audio with librosa
        audio, sr = librosa.load(
            audio_path, 
            sr=self.sample_rate,
            mono=(self.n_channels == 1)
        )
        
        # Handle channel dimension
        if self.n_channels == 1:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)  # Convert to mono
        elif self.n_channels == 2:
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)  # Duplicate for stereo
        
        # Normalize if requested
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Ensure float32 dtype
        audio = audio.astype(np.float32, copy=False)
        return audio
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio signal
            
        Returns:
            STFT magnitude spectrogram
        """
        # Compute STFT using librosa
        # Compute STFT using librosa; exclude Nyquist by slicing later to mirror madmom bins
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=np.hanning(self.n_fft).astype(np.float32),
            center=True,
            pad_mode='constant'
        )
        
        # Take magnitude and exclude Nyquist bin to align with madmom's fft_frequencies
        magnitude = np.abs(stft)[: self.n_fft // 2, :].astype(np.float32, copy=False)
        
        return magnitude
    
    def apply_filterbank(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply logarithmic filterbank to spectrogram.
        
        Args:
            spectrogram: STFT magnitude spectrogram [n_fft_bins, n_frames]
            
        Returns:
            Filtered spectrogram [n_filter_bins, n_frames]
        """
        # Apply filterbank: [n_filters, n_fft_bins] @ [n_fft_bins, n_frames]
        filtered = (self.filterbank @ spectrogram.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        
        # Apply logarithmic scaling similar to madmom: log10(1 + x)
        # Inputs are non-negative due to magnitude STFT and non-negative filters
        filtered = np.log10(1.0 + filtered).astype(np.float32, copy=False)
        
        return filtered
    
    def process_audio(self, audio_path: str) -> np.ndarray:
        """
        Complete audio processing pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed spectrogram of shape [n_frames, n_bins, n_channels]
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Handle multi-channel processing
        if self.n_channels == 1:
            # Mono processing
            stft = self.compute_stft(audio)
            filtered = self.apply_filterbank(stft)
            
            # Transpose to [n_frames, n_bins] and add channel dimension
            result = filtered.T.astype(np.float32, copy=False)  # [n_frames, n_bins]
            result = result[:, :, np.newaxis]  # [n_frames, n_bins, 1]
            
        else:
            # Stereo processing
            results = []
            for ch in range(self.n_channels):
                stft = self.compute_stft(audio[ch])
                filtered = self.apply_filterbank(stft)
                results.append(filtered.T)  # [n_frames, n_bins]
            
            # Stack channels: [n_frames, n_bins, n_channels]
            result = np.stack(results, axis=2).astype(np.float32, copy=False)
        
        return result
    
    def get_n_bins(self) -> int:
        """Get number of frequency bins after filterbank processing."""
        return self.n_bins


def create_adtof_processor(**kwargs) -> AudioProcessor:
    """
    Create AudioProcessor with ADTOF default parameters.
    
    Args:
        **kwargs: Override default parameters
        
    Returns:
        Configured AudioProcessor
    """
    defaults = {
        'sample_rate': 44100,
        'fps': 100,
        'frame_size': 2048,
        'bands_per_octave': 12,
        'fmin': 20.0,
        'fmax': 20000.0,
        'n_channels': 1,
        'normalize': False
    }
    
    # Update with any provided overrides
    defaults.update(kwargs)
    
    return AudioProcessor(**defaults)


def process_audio_file(audio_path: str, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Convenience function to process an audio file with ADTOF parameters.
    
    Args:
        audio_path: Path to audio file
        **kwargs: AudioProcessor parameters
        
    Returns:
        Tuple of (processed_spectrogram, n_bins)
    """
    processor = create_adtof_processor(**kwargs)
    spectrogram = processor.process_audio(audio_path)
    
    return spectrogram, processor.get_n_bins()


def compare_with_madmom(audio_path: str, use_madmom: bool = True) -> None:
    """
    Compare our implementation with madmom (if available).
    
    Args:
        audio_path: Path to test audio file
        use_madmom: Whether to load madmom for comparison
    """
    print("Comparing audio processing implementations...")
    
    # Our implementation
    print("\n1. Clean implementation:")
    try:
        spectrogram_clean, n_bins_clean = process_audio_file(audio_path)
        print(f"   Shape: {spectrogram_clean.shape}")
        print(f"   N bins: {n_bins_clean}")
        print(f"   Range: [{spectrogram_clean.min():.3f}, {spectrogram_clean.max():.3f}]")
        print(f"   Dtype: {spectrogram_clean.dtype}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Madmom implementation (if available)
    if use_madmom:
        print("\n2. Madmom implementation:")
        try:
            from adtof.io.mir import preProcess, getDim
            
            spectrogram_madmom = preProcess(audio_path)
            n_bins_madmom = getDim()
            
            print(f"   Shape: {spectrogram_madmom.shape}")
            print(f"   N bins: {n_bins_madmom}")
            print(f"   Range: [{spectrogram_madmom.min():.3f}, {spectrogram_madmom.max():.3f}]")
            print(f"   Dtype: {spectrogram_madmom.dtype}")
            
            # Compare shapes and values
            print("\n3. Comparison:")
            print(f"   Shape match: {spectrogram_clean.shape == spectrogram_madmom.shape}")
            print(f"   N bins match: {n_bins_clean == n_bins_madmom}")
            
            if spectrogram_clean.shape == spectrogram_madmom.shape:
                mse = np.mean((spectrogram_clean - spectrogram_madmom) ** 2)
                max_diff = np.max(np.abs(spectrogram_clean - spectrogram_madmom))
                print(f"   MSE: {mse:.6f}")
                print(f"   Max diff: {max_diff:.6f}")
            
        except ImportError:
            print("   Madmom not available for comparison")
        except Exception as e:
            print(f"   Error with madmom: {e}")


if __name__ == "__main__":
    # Test the implementation
    audio_path = "test.wav"
    
    print("Testing clean audio processing implementation...")
    print(f"Processing: {audio_path}")
    
    try:
        # Test our implementation
        spectrogram, n_bins = process_audio_file(audio_path)
        
        print("\n✓ Processing successful!")
        print(f"  Input file: {audio_path}")
        print(f"  Output shape: {spectrogram.shape}")
        print(f"  Frequency bins: {n_bins}")
        print(f"  Value range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]")
        print(f"  Duration: {spectrogram.shape[0] / 100:.1f} seconds (at 100 FPS)")
        
        # Compare with madmom if available
        compare_with_madmom(audio_path, use_madmom=True)
        
    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
