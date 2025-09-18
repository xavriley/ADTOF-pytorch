"""
PyTorch implementation of ADTOF Frame_RNN model and helpers.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .audio import create_adtof_processor, process_audio_file


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
        freq = x.shape[-1]
        out_w = (freq + self.stride_width - 1) // self.stride_width
        pad_needed = max(0, (out_w - 1) * self.stride_width + self.kernel_width - freq)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_left or pad_right:
            x = torch.nn.functional.pad(x, (pad_left, pad_right, 0, 0))
        return self.pool(x)


class KerasGRUCell(nn.Module):
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
        for p in [self.weight_ih, self.weight_hh]:
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        H = self.hidden_size
        W_ir, W_iz, W_in = self.weight_ih[:H], self.weight_ih[H:2 * H], self.weight_ih[2 * H:]
        W_hr, W_hz, W_hn = self.weight_hh[:H], self.weight_hh[H:2 * H], self.weight_hh[2 * H:]
        b_ir, b_iz, b_in = self.bias_ih[:H], self.bias_ih[H:2 * H], self.bias_ih[2 * H:]
        b_hr, b_hz, b_hn = self.bias_hh[:H], self.bias_hh[H:2 * H], self.bias_hh[2 * H:]

        r = torch.sigmoid(x_t @ W_ir.T + b_ir + h_prev @ W_hr.T + b_hr)
        z = torch.sigmoid(x_t @ W_iz.T + b_iz + h_prev @ W_hz.T + b_hz)
        n_pre = x_t @ W_in.T + b_in + r * (h_prev @ W_hn.T + b_hn)
        n = torch.tanh(n_pre)
        h_t = (1.0 - z) * n + z * h_prev
        return h_t


class KerasGRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        assert bidirectional, "KerasGRULayer currently supports only bidirectional=True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fw = KerasGRUCell(input_size, hidden_size)
        self.register_parameter('weight_ih_l0', self.fw.weight_ih)
        self.register_parameter('weight_hh_l0', self.fw.weight_hh)
        self.register_parameter('bias_ih_l0', self.fw.bias_ih)
        self.register_parameter('bias_hh_l0', self.fw.bias_hh)
        self.bw = KerasGRUCell(input_size, hidden_size)
        self.register_parameter('weight_ih_l0_reverse', self.bw.weight_ih)
        self.register_parameter('weight_hh_l0_reverse', self.bw.weight_hh)
        self.register_parameter('bias_ih_l0_reverse', self.bw.bias_ih)
        self.register_parameter('bias_hh_l0_reverse', self.bw.bias_hh)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, I = x.shape
        H = self.hidden_size
        h_fw = x.new_zeros(B, H)
        outs_fw = []
        for t in range(T):
            h_fw = self.fw(x[:, t, :], h_fw)
            outs_fw.append(h_fw)
        y_fw = torch.stack(outs_fw, dim=1)
        h_bw = x.new_zeros(B, H)
        outs_bw = []
        for t in reversed(range(T)):
            h_bw = self.bw(x[:, t, :], h_bw)
            outs_bw.append(h_bw)
        outs_bw.reverse()
        y_bw = torch.stack(outs_bw, dim=1)
        y = torch.cat([y_fw, y_bw], dim=2)
        return y, (h_fw, h_bw)


class ContextLayer(nn.Module):
    def __init__(self, context_frames: int):
        super().__init__()
        self.context_frames = context_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, features = x.shape
        to_concat = []
        for offset in range(self.context_frames - 1):
            end_idx = time_steps - (self.context_frames - offset - 1)
            to_concat.append(x[:, offset:end_idx, :])
        to_concat.append(x[:, (self.context_frames - 1):, :])
        return torch.cat(to_concat, dim=2)


class ADTOFFrameRNN(nn.Module):
    def __init__(
        self,
        n_bins: int = 168,
        n_channels: int = 1,
        conv_filters: List[int] = [32, 64],
        gru_units: List[int] = [60, 60, 60],
        context: int = 9,
        output_classes: int = 5,
        same_padding: bool = True,
        dropout_rate: float = 0.3,
        use_keras_gru: bool = False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.n_channels = n_channels
        self.context = context
        self.same_padding = same_padding
        self.conv_filters = conv_filters

        self.cnn_blocks = nn.ModuleList()
        in_channels = n_channels
        for i, filters in enumerate(conv_filters):
            block = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1 if same_padding else 0),
                nn.ReLU(),
                nn.BatchNorm2d(filters, eps=1e-3),
                nn.Conv2d(filters, filters, kernel_size=3, padding=1 if same_padding else 0),
                nn.ReLU(),
                nn.BatchNorm2d(filters, eps=1e-3),
                None,
                nn.Dropout2d(dropout_rate),
            )
            self.cnn_blocks.append(block)
            in_channels = filters

        for b in self.cnn_blocks:
            for idx, layer in enumerate(b):
                if layer is None:
                    b[idx] = SamePadWidthMaxPool(kernel_width=3)
                    break

        self.cnn_output_features = self._calculate_cnn_output_size()

        if context > 1:
            cnn_receptive_field = len(conv_filters) * (2 * 2) + 1
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

        rnn_input_size = self.cnn_output_features * self.context_multiplier
        self.gru_layers = nn.ModuleList()
        for i, units in enumerate(gru_units):
            input_size = rnn_input_size if i == 0 else gru_units[i - 1] * 2
            if use_keras_gru:
                self.gru_layers.append(KerasGRULayer(input_size, units, bidirectional=True))
            else:
                self.gru_layers.append(nn.GRU(input_size, units, batch_first=True, bidirectional=True))

        final_gru_size = gru_units[-1] * 2
        self.output_layer = nn.Linear(final_gru_size, output_classes)

    def _calculate_cnn_output_size(self) -> int:
        dummy_input = torch.randn(1, 1, 100, self.n_bins)
        with torch.no_grad():
            x = dummy_input
            for cnn_block in self.cnn_blocks:
                x = cnn_block(x)
            features_per_timestep = x.shape[1] * x.shape[3]
        return features_per_timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, freq_bins, channels = x.shape
        x = x.permute(0, 3, 1, 2)
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        x = x.permute(0, 2, 3, 1)
        features = x.shape[2] * x.shape[3]
        x = x.reshape(batch_size, time_steps, features)
        if hasattr(self, 'context_layer') and self.context_layer is not None:
            x = self.context_layer(x)
        for gru in self.gru_layers:
            x, _ = gru(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_output_features': self.cnn_output_features,
            'context_multiplier': getattr(self, 'context_multiplier', 1),
            'architecture': 'Frame_RNN',
        }


def create_frame_rnn_model(n_bins: int = 168) -> ADTOFFrameRNN:
    use_keras_gru_env = os.environ.get('ADTOF_USE_KERAS_GRU', '0')
    use_keras_gru = use_keras_gru_env.strip() in ('1', 'true', 'True')
    return ADTOFFrameRNN(
        n_bins=n_bins,
        n_channels=1,
        conv_filters=[32, 64],
        gru_units=[60, 60, 60],
        context=9,
        output_classes=5,
        same_padding=True,
        dropout_rate=0.3,
        use_keras_gru=use_keras_gru,
    )


def calculate_n_bins(bands_per_octave: int = 12, fmin: float = 20, fmax: float = 20000,
                     frame_size: int = 2048, sample_rate: int = 44100) -> int:
    processor = create_adtof_processor(
        sample_rate=sample_rate,
        frame_size=frame_size,
        bands_per_octave=bands_per_octave,
        fmin=fmin,
        fmax=fmax,
    )
    return processor.get_n_bins()


def load_audio_for_model(audio_path: str, **kwargs) -> torch.Tensor:
    spectrogram, n_bins = process_audio_file(audio_path, **kwargs)
    tensor = torch.from_numpy(spectrogram).float()
    tensor = tensor.unsqueeze(0)
    return tensor


def load_pytorch_weights(model: ADTOFFrameRNN, weights_path: str, strict: bool = False) -> ADTOFFrameRNN:
    print(f"Loading PyTorch weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_weights' in checkpoint:
        weights = checkpoint['model_weights']
    else:
        weights = checkpoint

    try:
        if hasattr(model, 'gru_layers'):
            for i, layer in enumerate(model.gru_layers):
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
                    for target_key, source_key in alias_map.items():
                        if source_key in weights and target_key not in weights:
                            weights[target_key] = weights[source_key]
    except Exception:
        pass

    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=strict)
    if not strict:
        if missing_keys:
            print(f"Missing keys (using random initialization): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
    return model
