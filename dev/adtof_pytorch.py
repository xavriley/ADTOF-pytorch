"""
PyTorch implementation of ADTOF Frame_RNN model for drum transcription.

This is a clean reimplementation of the TensorFlow model architecture:
- CNN: Conv2D + BatchNorm + MaxPool layers
- Context: Frame stacking for temporal context
- RNN: Bidirectional GRU layers  
- Output: Dense layer with sigmoid activation

Architecture matches the Frame_RNN hyperparameters from the original model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Optional
class SamePadWidthMaxPool(nn.Module):
    """
    MaxPool over width (frequency) with TensorFlow/Keras 'same' padding semantics.
    Operates on tensors shaped [batch, channels, time, freq].
    Uses kernel_size=(1, 3), stride=(1, 3).
    """
    def __init__(self, kernel_width: int = 3, stride_width: int = 3):
        super().__init__()
        self.kernel_width = kernel_width
        self.stride_width = stride_width
        self.pool = nn.MaxPool2d(kernel_size=(1, kernel_width), stride=(1, stride_width), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, F]
        freq = x.shape[-1]
        # Output width per TF 'same': ceil(F / stride)
        out_w = (freq + self.stride_width - 1) // self.stride_width
        pad_needed = max(0, (out_w - 1) * self.stride_width + self.kernel_width - freq)
        # TF/Keras 'SAME' padding splits padding between start/end (more on the end if odd)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_left or pad_right:
            # F.pad pads last two dims as (pad_left, pad_right, pad_top, pad_bottom)
            x = torch.nn.functional.pad(x, (pad_left, pad_right, 0, 0))
        return self.pool(x)


class KerasGRUCell(nn.Module):
    """
    GRU cell matching Keras equations with reset_after=True and activations tanh/sigmoid.
    Parameters are organized like PyTorch GRU: weight_ih [3H, I], weight_hh [3H, H],
    bias_ih [3H], bias_hh [3H], with gate order (r, z, n).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        # Use a simple Kaiming uniform-like init consistent with PyTorch defaults
        for p in [self.weight_ih, self.weight_hh]:
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        H = self.hidden_size
        # Slices
        W_ir, W_iz, W_in = self.weight_ih[:H], self.weight_ih[H:2*H], self.weight_ih[2*H:]
        W_hr, W_hz, W_hn = self.weight_hh[:H], self.weight_hh[H:2*H], self.weight_hh[2*H:]
        b_ir, b_iz, b_in = self.bias_ih[:H], self.bias_ih[H:2*H], self.bias_ih[2*H:]
        b_hr, b_hz, b_hn = self.bias_hh[:H], self.bias_hh[H:2*H], self.bias_hh[2*H:]

        # Gates: r, z (reset_after=True style)
        r = torch.sigmoid(x_t @ W_ir.T + b_ir + h_prev @ W_hr.T + b_hr)
        z = torch.sigmoid(x_t @ W_iz.T + b_iz + h_prev @ W_hz.T + b_hz)
        # Candidate uses reset gate on recurrent contribution and separate recurrent bias for n
        n_pre = x_t @ W_in.T + b_in + r * (h_prev @ W_hn.T + b_hn)
        n = torch.tanh(n_pre)
        h_t = (1.0 - z) * n + z * h_prev
        return h_t


class KerasGRULayer(nn.Module):
    """
    Bidirectional GRU layer built from two KerasGRUCell instances, exposing PyTorch GRU parameter names
    so the existing weight converter/state_dict mapping remains valid.
    """
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        assert bidirectional, "KerasGRULayer currently supports only bidirectional=True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Forward cell
        self.fw = KerasGRUCell(input_size, hidden_size)
        # Register with PyTorch GRU-compatible names
        self.register_parameter('weight_ih_l0', self.fw.weight_ih)
        self.register_parameter('weight_hh_l0', self.fw.weight_hh)
        self.register_parameter('bias_ih_l0', self.fw.bias_ih)
        self.register_parameter('bias_hh_l0', self.fw.bias_hh)
        # Backward cell
        self.bw = KerasGRUCell(input_size, hidden_size)
        self.register_parameter('weight_ih_l0_reverse', self.bw.weight_ih)
        self.register_parameter('weight_hh_l0_reverse', self.bw.weight_hh)
        self.register_parameter('bias_ih_l0_reverse', self.bw.bias_ih)
        self.register_parameter('bias_hh_l0_reverse', self.bw.bias_hh)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, T, I]
        B, T, I = x.shape
        H = self.hidden_size
        # Forward pass
        h_fw = x.new_zeros(B, H)
        outs_fw = []
        for t in range(T):
            h_fw = self.fw(x[:, t, :], h_fw)
            outs_fw.append(h_fw)
        y_fw = torch.stack(outs_fw, dim=1)  # [B, T, H]
        # Backward pass
        h_bw = x.new_zeros(B, H)
        outs_bw = []
        for t in reversed(range(T)):
            h_bw = self.bw(x[:, t, :], h_bw)
            outs_bw.append(h_bw)
        outs_bw.reverse()
        y_bw = torch.stack(outs_bw, dim=1)  # [B, T, H]
        # Concatenate outputs
        y = torch.cat([y_fw, y_bw], dim=2)  # [B, T, 2H]
        # Return output and tuple of final hidden states (fw, bw)
        return y, (h_fw, h_bw)


class ContextLayer(nn.Module):
    """
    Adds temporal context by stacking adjacent frames.
    Equivalent to the _add_context function in the TensorFlow version.
    """
    def __init__(self, context_frames: int):
        super().__init__()
        self.context_frames = context_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, time, features]
        Returns:
            Tensor with context frames concatenated: [batch, time-context+1, features*context]
        """
        batch_size, time_steps, features = x.shape
        
        # Create list of shifted versions for context
        to_concat = []
        for offset in range(self.context_frames - 1):
            end_idx = time_steps - (self.context_frames - offset - 1)
            to_concat.append(x[:, offset:end_idx, :])
        
        # Add the final slice
        to_concat.append(x[:, (self.context_frames - 1):, :])
        
        # Concatenate along feature dimension
        return torch.cat(to_concat, dim=2)


class ADTOFFrameRNN(nn.Module):
    """
    PyTorch implementation of ADTOF Frame_RNN model.
    
    Architecture:
    1. CNN: 2 blocks of (Conv2D + BatchNorm + Conv2D + BatchNorm + MaxPool + Dropout)
    2. Context: Frame stacking for temporal receptive field
    3. RNN: 3 Bidirectional GRU layers
    4. Output: Dense layer with sigmoid activation
    """
    
    def __init__(
        self,
        n_bins: int = 168,  # Frequency bins (calculated from audio processing)
        n_channels: int = 1,  # Mono audio
        conv_filters: List[int] = [32, 64],  # CNN filter sizes
        gru_units: List[int] = [60, 60, 60],  # GRU hidden sizes
        context: int = 9,  # Context frames
        output_classes: int = 5,  # BD, SD, HH, TT, CY
        same_padding: bool = True,
        dropout_rate: float = 0.3,
        use_keras_gru: bool = False
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_channels = n_channels
        self.context = context
        self.same_padding = same_padding
        self.conv_filters = conv_filters
        
        # CNN layers
        self.cnn_blocks = nn.ModuleList()
        in_channels = n_channels
        
        for i, filters in enumerate(conv_filters):
            block = nn.Sequential(
                # First conv block: Conv -> ReLU -> BN (match Keras Conv with activation followed by BN)
                nn.Conv2d(in_channels, filters, kernel_size=3,
                         padding=1 if same_padding else 0),
                nn.ReLU(),
                nn.BatchNorm2d(filters, eps=1e-3),
                
                # Second conv block: Conv -> ReLU -> BN
                nn.Conv2d(filters, filters, kernel_size=3,
                         padding=1 if same_padding else 0),
                nn.ReLU(), 
                nn.BatchNorm2d(filters, eps=1e-3),
                
                # Max pooling (only in frequency dimension)  
                # TensorFlow uses 'same' padding for pooling: pad only at end to keep ceil division
                # We mimic this with a custom pooling that right-pads width as needed
                None,  # placeholder to be replaced below
                nn.Dropout2d(dropout_rate)
            )
            self.cnn_blocks.append(block)
            in_channels = filters

        # Replace the placeholder with a custom same-padding pooling module
        for b in self.cnn_blocks:
            # Find index of placeholder (None) which should be at position 6
            for idx, layer in enumerate(b):
                if layer is None:
                    b[idx] = SamePadWidthMaxPool(kernel_width=3)
                    break
        
        # Calculate CNN output dimensions
        self.cnn_output_features = self._calculate_cnn_output_size()
        
        # Context layer
        if context > 1:
            # Calculate context frames needed beyond CNN receptive field
            cnn_receptive_field = len(conv_filters) * (2 * 2) + 1  # 9 for 2 blocks
            context_frames = context - cnn_receptive_field + 1
            if context_frames > 1:
                self.context_layer = ContextLayer(context_frames)
                self.context_multiplier = context_frames
            else:
                self.context_layer = None
                self.context_multiplier = 1
        else:
            self.context_layer = None
            self.context_multiplier = 1
        
        # RNN layers
        rnn_input_size = self.cnn_output_features * self.context_multiplier
        self.gru_layers = nn.ModuleList()
        
        for i, units in enumerate(gru_units):
            input_size = rnn_input_size if i == 0 else gru_units[i-1] * 2  # *2 for bidirectional
            if use_keras_gru:
                self.gru_layers.append(KerasGRULayer(input_size, units, bidirectional=True))
            else:
                self.gru_layers.append(
                    nn.GRU(input_size, units, batch_first=True, bidirectional=True)
                )
        
        # Output layer
        final_gru_size = gru_units[-1] * 2  # *2 for bidirectional
        self.output_layer = nn.Linear(final_gru_size, output_classes)
        
    def _calculate_cnn_output_size(self) -> int:
        """Calculate the number of features after CNN processing by running a dummy forward pass."""
        # Create a dummy input matching the actual input format
        dummy_input = torch.randn(1, 1, 100, self.n_bins)  # [batch, channels, time, freq]
        
        with torch.no_grad():
            x = dummy_input
            for cnn_block in self.cnn_blocks:
                x = cnn_block(x)
            
            # x is now [batch, channels, time, freq] -> calculate features per timestep
            features_per_timestep = x.shape[1] * x.shape[3]  # channels * freq
        
        return features_per_timestep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch, time, freq, channels]
            
        Returns:
            Output predictions of shape [batch, time, classes]
        """
        batch_size, time_steps, freq_bins, channels = x.shape
        
        # Reshape for CNN: [batch, channels, time, freq] for 2D convolution
        # TensorFlow processes as (batch, time, freq, channels) -> need to match this
        x = x.permute(0, 3, 1, 2)  # [batch, channels, time, freq]
        
        # CNN processing
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # Reshape back to sequence: [batch, time, features]
        # Match Keras: channels-last before flatten, flatten last two dims (freq, channels)
        x = x.permute(0, 2, 3, 1)  # [batch, time, freq, channels]
        features = x.shape[2] * x.shape[3]  # freq * channels
        x = x.reshape(batch_size, time_steps, features)
        
        # Apply context if needed
        if self.context_layer is not None:
            x = self.context_layer(x)
        
        # RNN processing
        for gru in self.gru_layers:
            x, _ = gru(x)
        
        # Output layer with sigmoid activation
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Return model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_output_features': self.cnn_output_features,
            'context_multiplier': self.context_multiplier,
            'architecture': 'Frame_RNN'
        }


def create_frame_rnn_model(n_bins: int = 168) -> ADTOFFrameRNN:
    """
    Create Frame_RNN model with default hyperparameters matching TensorFlow version.
    
    Args:
        n_bins: Number of frequency bins (default calculated for 12 bands/octave, 20-20kHz)
    
    Returns:
        Initialized ADTOFFrameRNN model
    """
    use_keras_gru_env = os.environ.get('ADTOF_USE_KERAS_GRU', '1')
    use_keras_gru = use_keras_gru_env.strip() in ('1', 'true', 'True')
    return ADTOFFrameRNN(
        n_bins=n_bins,
        n_channels=1,
        conv_filters=[32, 64],  # From Frame_RNN hyperparameters
        gru_units=[60, 60, 60],  # From Frame_RNN hyperparameters  
        context=9,  # From Frame_RNN hyperparameters
        output_classes=5,  # BD, SD, HH, TT, CY
        same_padding=True,  # From Frame_RNN hyperparameters
        dropout_rate=0.3,
        use_keras_gru=use_keras_gru
    )


def calculate_n_bins(bands_per_octave: int = 12, fmin: float = 20, fmax: float = 20000,
                    frame_size: int = 2048, sample_rate: int = 44100) -> int:
    """
    Calculate number of frequency bins using same logic as TensorFlow version.
    This matches the getDim function in adtof/io/mir.py
    """
    from audio_processing import create_adtof_processor
    
    processor = create_adtof_processor(
        sample_rate=sample_rate,
        frame_size=frame_size,
        bands_per_octave=bands_per_octave,
        fmin=fmin,
        fmax=fmax
    )
    return processor.get_n_bins()


def load_audio_for_model(audio_path: str, **kwargs) -> torch.Tensor:
    """
    Load and process audio file for ADTOF model inference.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Audio processing parameters
        
    Returns:
        Processed audio tensor ready for model input [1, time, freq_bins, 1]
    """
    # from adtof.io.mir import preProcess
    from audio_processing import process_audio_file
    
    # Process audio with ADTOF parameters
    spectrogram, n_bins = process_audio_file(audio_path, **kwargs)
    # print(f"Spectrogram original shape: {spectrogram.shape}")

    # spectrogram = preProcess(audio_path)
    # print(f"Spectrogram madmom shape: {spectrogram.shape}")

    # Convert to PyTorch tensor and add batch dimension
    tensor = torch.from_numpy(spectrogram).float()
    tensor = tensor.unsqueeze(0)  # Add batch dimension: [1, time, freq_bins, channels]
    
    return tensor


def load_pytorch_weights(model: ADTOFFrameRNN, weights_path: str, strict: bool = False) -> ADTOFFrameRNN:
    """
    Load pre-converted PyTorch weights into the model.
    
    Args:
        model: PyTorch model instance
        weights_path: Path to saved PyTorch weights (.pth file)
        strict: Whether to strictly enforce all keys match
    
    Returns:
        Model with loaded weights
    """
    print(f"Loading PyTorch weights from: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'model_weights' in checkpoint:
        weights = checkpoint['model_weights']
        info = checkpoint.get('model_info', {})
        print(f"Loading weights for: {info.get('architecture', 'Unknown')}")
    else:
        weights = checkpoint

    # If using the KerasGRU variant, pre-populate alias keys (fw/bw.*) from
    # the PyTorch-GRU style keys (*_l0 and *_l0_reverse) to avoid benign
    # "missing keys" warnings. This keeps a single source of truth while
    # satisfying both name sets exposed by KerasGRULayer.
    try:
        if hasattr(model, 'gru_layers'):
            for i, layer in enumerate(model.gru_layers):
                # Only applies to KerasGRULayer which exposes fw/bw submodules
                if hasattr(layer, 'fw') and hasattr(layer, 'bw'):
                    prefix = f"gru_layers.{i}"
                    alias_map = {
                        f"{prefix}.fw.weight_ih": f"{prefix}.weight_ih_l0",
                        f"{prefix}.fw.weight_hh": f"{prefix}.weight_hh_l0",
                        f"{prefix}.fw.bias_ih": f"{prefix}.bias_ih_l0",
                        f"{prefix}.fw.bias_hh": f"{prefix}.bias_hh_l0",
                        f"{prefix}.bw.weight_ih": f"{prefix}.weight_ih_l0_reverse",
                        f"{prefix}.bw.weight_hh": f"{prefix}.weight_hh_l0_reverse",
                        f"{prefix}.bw.bias_ih": f"{prefix}.bias_ih_l0_reverse",
                        f"{prefix}.bw.bias_hh": f"{prefix}.bias_hh_l0_reverse",
                    }
                    # Populate any missing alias from its source key
                    for target_key, source_key in alias_map.items():
                        if source_key in weights and target_key not in weights:
                            weights[target_key] = weights[source_key]
    except Exception:
        # Do not fail weight loading due to alias population issues
        pass
    
    # Load weights with optional strict mode
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=strict)
    
    if not strict:
        if missing_keys:
            print(f"Missing keys (using random initialization): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
    
    print("Weights loaded successfully!")
    return model


if __name__ == "__main__":
    # Test the model
    print("Creating ADTOF Frame_RNN PyTorch model...")
    
    # Calculate input dimensions
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    
    print(f"Model created with {n_bins} frequency bins")
    print("Model info:", model.get_model_info())
    
    # Test with dummy input
    batch_size, time_steps = 2, 400
    dummy_input = torch.randn(batch_size, time_steps, n_bins, 1)
    
    print(f"Testing with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("Model test completed successfully!")
    
    # Test loading PyTorch weights if available
    weights_path = "adtof_frame_rnn_pytorch_weights.pth"
    if os.path.exists(weights_path):
        print(f"\nFound PyTorch weights file: {weights_path}")
        try:
            model_with_weights = load_pytorch_weights(model, weights_path, strict=False)
            print("PyTorch weights loaded successfully!")
            
            # Test with loaded weights
            with torch.no_grad():
                output_with_weights = model_with_weights(dummy_input)
            print(f"Output with loaded weights: {output_with_weights.shape}")
            print(f"Range: [{output_with_weights.min().item():.3f}, {output_with_weights.max().item():.3f}]")
            
        except Exception as e:
            print(f"Failed to load PyTorch weights: {e}")
    else:
        print(f"\nNo PyTorch weights found at: {weights_path}")
        print("Run 'python convert_weights.py' to convert from TensorFlow model")
