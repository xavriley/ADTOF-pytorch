"""
Weight conversion utility for ADTOF Frame_RNN model.
Converts TensorFlow checkpoint weights to PyTorch format and saves them.

Usage:
    python convert_weights.py
"""

import torch
import numpy as np
import os
from typing import Dict
from adtof_pytorch import create_frame_rnn_model, ADTOFFrameRNN

# Use double precision in PyTorch for tighter numerical parity
torch.set_default_dtype(torch.float64)


def analyze_tensorflow_model_structure(tf_model):
    """
    Analyze TensorFlow model structure to understand weight organization.
    """
    print("=== TensorFlow Model Analysis ===")
    
    weights = tf_model.model.get_weights()
    layer_names = [layer.name for layer in tf_model.model.layers]
    
    print(f"Total weight tensors: {len(weights)}")
    print(f"Total layers: {len(tf_model.model.layers)}")
    
    print("\nLayer structure:")
    for i, layer in enumerate(tf_model.model.layers):
        ltype = type(layer).__name__
        print(f"  {i}: {layer.name} ({ltype})")
        # Print key configs for parity checks
        try:
            cfg = layer.get_config()
        except Exception:
            cfg = {}
        if 'Conv2D' in ltype:
            act = cfg.get('activation', None)
            use_bias = cfg.get('use_bias', None)
            padding = cfg.get('padding', None)
            print(f"      Conv2D cfg: activation={act}, use_bias={use_bias}, padding={padding}")
        if 'BatchNormalization' in ltype:
            eps = cfg.get('epsilon', None)
            mom = cfg.get('momentum', None)
            center = cfg.get('center', None)
            scale = cfg.get('scale', None)
            axis = cfg.get('axis', None)
            print(f"      BN cfg: epsilon={eps}, momentum={mom}, center={center}, scale={scale}, axis={axis}")
        if 'Bidirectional' in ltype:
            inner = getattr(layer, 'layer', None)
            inner_type = type(inner).__name__ if inner is not None else None
            inner_cfg = inner.get_config() if inner is not None and hasattr(inner, 'get_config') else {}
            ra = inner_cfg.get('reset_after', None)
            act = inner_cfg.get('activation', None)
            ract = inner_cfg.get('recurrent_activation', None)
            use_bias = inner_cfg.get('use_bias', None)
            units = inner_cfg.get('units', None)
            dropout = inner_cfg.get('dropout', None)
            rec_dropout = inner_cfg.get('recurrent_dropout', None)
            print(f"      Bi{inner_type} cfg: units={units}, reset_after={ra}, act={act}, r_act={ract}, use_bias={use_bias}, drop={dropout}, r_drop={rec_dropout}")
        if hasattr(layer, 'get_weights') and layer.get_weights():
            layer_weights = layer.get_weights()
            print(f"      Weights: {[w.shape for w in layer_weights]}")
    
    print("\nAll weight shapes:")
    for i, w in enumerate(weights):
        print(f"  {i}: {w.shape}")
    
    return weights, layer_names


def extract_cnn_weights(tf_model, pytorch_model: ADTOFFrameRNN) -> Dict[str, torch.Tensor]:
    """
    Extract CNN weights from TensorFlow model and convert to PyTorch format.
    """
    print("\n=== Extracting CNN Weights ===")
    
    cnn_weights = {}
    
    # Build target parameter slots for PyTorch CNN blocks
    conv_targets = []
    bn_targets = []
    for block_idx in range(len(pytorch_model.conv_filters)):
        # Conv layers at indices 0 and 3
        conv_targets.append({
            'w': f"cnn_blocks.{block_idx}.0.weight",
            'b': f"cnn_blocks.{block_idx}.0.bias"
        })
        conv_targets.append({
            'w': f"cnn_blocks.{block_idx}.3.weight",
            'b': f"cnn_blocks.{block_idx}.3.bias"
        })
        # BatchNorm layers at indices 2 and 5 (Conv -> ReLU -> BN)
        bn_targets.append(f"cnn_blocks.{block_idx}.2")
        bn_targets.append(f"cnn_blocks.{block_idx}.5")
    
    conv_assigned = 0
    bn_assigned = 0
    
    # Helper: recursively flatten Keras layers to handle wrappers (e.g., TimeDistributed)
    def _flatten_layers(layer, acc):
        acc.append(layer)
        # If this is a nested Model-like layer
        if hasattr(layer, 'layers') and isinstance(getattr(layer, 'layers'), (list, tuple)):
            for sub in layer.layers:
                _flatten_layers(sub, acc)
        # If this is a wrapper (e.g., TimeDistributed) exposing a single sub-layer
        if hasattr(layer, 'layer'):
            _flatten_layers(layer.layer, acc)
        return acc

    flat_layers = []
    for top in tf_model.model.layers:
        _flatten_layers(top, flat_layers)

    # Iterate flattened layers and map Conv2D/BatchNormalization
    for layer in flat_layers:
        ltype = type(layer).__name__.lower()
        if 'sequential' in ltype or 'model' in ltype or 'inputlayer' in ltype:
            continue
        if 'conv2d' in ltype:
            lw = layer.get_weights()
            if not lw:
                continue
            kernel = lw[0]
            if len(lw) > 1 and lw[1].ndim == 1:
                bias_np = lw[1]
            else:
                # Keras conv with use_bias=False → create zero bias for PyTorch
                bias_np = np.zeros(kernel.shape[-1], dtype=kernel.dtype)
            # Convert to PyTorch format [out_channels, in_channels, H, W]
            pytorch_weight = torch.from_numpy(kernel.transpose(3, 2, 0, 1)).double()
            pytorch_bias = torch.from_numpy(bias_np).double()
            
            if conv_assigned >= len(conv_targets):
                print("Warning: More Conv2D layers in TF model than expected by PyTorch model")
                continue
            target = conv_targets[conv_assigned]
            cnn_weights[target['w']] = pytorch_weight
            cnn_weights[target['b']] = pytorch_bias
            print("  Mapped Conv2D", conv_assigned, "->", target['w'], f"{kernel.shape} -> {pytorch_weight.shape}")
            conv_assigned += 1
        elif 'batchnormalization' in ltype:
            lw = layer.get_weights()
            if not lw:
                continue
            # Expect [gamma, beta, moving_mean, moving_variance]
            if len(lw) < 4:
                continue
            gamma = lw[0]
            beta = lw[1]
            moving_mean = lw[2]
            moving_var = lw[3]
            if bn_assigned >= len(bn_targets):
                print("Warning: More BatchNorm layers in TF model than expected by PyTorch model")
                continue
            target_prefix = bn_targets[bn_assigned]
            cnn_weights[f"{target_prefix}.weight"] = torch.from_numpy(gamma).double()
            cnn_weights[f"{target_prefix}.bias"] = torch.from_numpy(beta).double()
            cnn_weights[f"{target_prefix}.running_mean"] = torch.from_numpy(moving_mean).double()
            cnn_weights[f"{target_prefix}.running_var"] = torch.from_numpy(moving_var).double()
            print("  Mapped BatchNorm", bn_assigned, "->", target_prefix, f"{gamma.shape}")
            bn_assigned += 1
        else:
            continue
    
    expected_convs = len(pytorch_model.conv_filters) * 2
    expected_bns = len(pytorch_model.conv_filters) * 2
    if conv_assigned != expected_convs or bn_assigned != expected_bns:
        print(f"Warning: Assigned {conv_assigned}/{expected_convs} Conv2D and {bn_assigned}/{expected_bns} BatchNorm via layer-walk; attempting shape-based fallback")
        # Fallback: shape-driven scan over flattened weights
        model_weights = tf_model.model.get_weights()
        conv_assigned_fb = 0
        bn_assigned_fb = 0
        idx = 0
        while idx < len(model_weights) and (conv_assigned_fb < expected_convs or bn_assigned_fb < expected_bns):
            w = model_weights[idx]
            # Conv kernel is 4D
            if hasattr(w, 'ndim') and w.ndim == 4 and conv_assigned_fb < expected_convs:
                kernel = w  # [H, W, inC, outC]
                out_c = kernel.shape[-1]
                # Optional bias next
                bias_np = None
                if idx + 1 < len(model_weights):
                    wnext = model_weights[idx + 1]
                    if hasattr(wnext, 'ndim') and wnext.ndim == 1 and wnext.shape[0] == out_c:
                        bias_np = wnext
                        idx += 1
                if bias_np is None:
                    bias_np = np.zeros(out_c, dtype=kernel.dtype)
                pyt_weight = torch.from_numpy(kernel.transpose(3, 2, 0, 1)).double()
                pyt_bias = torch.from_numpy(bias_np).double()
                if conv_assigned_fb < len(conv_targets):
                    target = conv_targets[conv_assigned_fb]
                    cnn_weights[target['w']] = pyt_weight
                    cnn_weights[target['b']] = pyt_bias
                    print("  [FB] Mapped Conv2D", conv_assigned_fb, "->", target['w'], f"{kernel.shape} -> {pyt_weight.shape}")
                conv_assigned_fb += 1
                idx += 1
                continue
            # BN set: expect gamma,beta,mean,var as 1D vectors
            if (idx + 3 < len(model_weights) and
                all(hasattr(model_weights[idx + k], 'ndim') and model_weights[idx + k].ndim == 1 for k in range(4)) and
                bn_assigned_fb < expected_bns):
                gamma = model_weights[idx]
                beta = model_weights[idx + 1]
                moving_mean = model_weights[idx + 2]
                moving_var = model_weights[idx + 3]
                if bn_assigned_fb < len(bn_targets):
                    prefix = bn_targets[bn_assigned_fb]
                    cnn_weights[f"{prefix}.weight"] = torch.from_numpy(gamma).double()
                    cnn_weights[f"{prefix}.bias"] = torch.from_numpy(beta).double()
                    cnn_weights[f"{prefix}.running_mean"] = torch.from_numpy(moving_mean).double()
                    cnn_weights[f"{prefix}.running_var"] = torch.from_numpy(moving_var).double()
                    print("  [FB] Mapped BatchNorm", bn_assigned_fb, "->", prefix, f"{gamma.shape}")
                bn_assigned_fb += 1
                idx += 4
                continue
            idx += 1
        print(f"Fallback assigned Conv2D: {conv_assigned_fb}/{expected_convs}, BatchNorm: {bn_assigned_fb}/{expected_bns}")
    
    return cnn_weights


def extract_gru_weights(tf_model, pytorch_model: ADTOFFrameRNN) -> Dict[str, torch.Tensor]:
    """
    Extract GRU weights from TensorFlow model and convert to PyTorch format.
    
    TensorFlow Bidirectional GRU structure:
    - Forward GRU: [(input_size, 3*hidden), (hidden, 3*hidden), (2, 3*hidden)]
    - Backward GRU: [(input_size, 3*hidden), (hidden, 3*hidden), (2, 3*hidden)]
    
    PyTorch Bidirectional GRU structure:
    - weight_ih_l0: [3*hidden, input_size] (input-to-hidden forward)
    - weight_hh_l0: [3*hidden, hidden] (hidden-to-hidden forward)  
    - bias_ih_l0: [3*hidden] (input bias forward)
    - bias_hh_l0: [3*hidden] (hidden bias forward)
    - weight_ih_l0_reverse: [3*hidden, input_size] (input-to-hidden backward)
    - weight_hh_l0_reverse: [3*hidden, hidden] (hidden-to-hidden backward)
    - bias_ih_l0_reverse: [3*hidden] (input bias backward)
    - bias_hh_l0_reverse: [3*hidden] (hidden bias backward)
    """
    print("\n=== Extracting GRU Weights ===")
    
    gru_weights = {}
    
    # Find bidirectional GRU layers
    bidirectional_layers = []
    for layer in tf_model.model.layers:
        if 'bidirectional' in layer.name.lower():
            bidirectional_layers.append(layer)
    
    print(f"Found {len(bidirectional_layers)} bidirectional GRU layers")
    
    for i, tf_layer in enumerate(bidirectional_layers):
        print(f"\nProcessing GRU layer {i}: {tf_layer.name}")
        
        # Get TensorFlow weights
        layer_weights = tf_layer.get_weights()
        print(f"  TF layer has {len(layer_weights)} weight tensors")
        
        if len(layer_weights) != 6:
            print(f"  ERROR: Expected 6 weight tensors, got {len(layer_weights)}")
            continue
        
        # TensorFlow weight organization:
        # 0: forward input-to-hidden weights [input_size, 3*hidden]
        # 1: forward hidden-to-hidden weights [hidden, 3*hidden] 
        # 2: forward biases [2, 3*hidden]
        # 3: backward input-to-hidden weights [input_size, 3*hidden]
        # 4: backward hidden-to-hidden weights [hidden, 3*hidden]
        # 5: backward biases [2, 3*hidden]
        
        tf_fw_ih = layer_weights[0]  # [input_size, 3*hidden]
        tf_fw_hh = layer_weights[1]  # [hidden, 3*hidden]
        tf_fw_bias = layer_weights[2]  # [2, 3*hidden] or [3*hidden]
        tf_bw_ih = layer_weights[3]  # [input_size, 3*hidden]  
        tf_bw_hh = layer_weights[4]  # [hidden, 3*hidden]
        tf_bw_bias = layer_weights[5]  # [2, 3*hidden] or [3*hidden]
        
        print(f"    Forward weights: ih={tf_fw_ih.shape}, hh={tf_fw_hh.shape}, bias={tf_fw_bias.shape}")
        print(f"    Backward weights: ih={tf_bw_ih.shape}, hh={tf_bw_hh.shape}, bias={tf_bw_bias.shape}")
        
        # Helper to reorder gates from Keras (z, r, h) to PyTorch (r, z, n)
        def reorder_kernel(mat: np.ndarray) -> np.ndarray:
            # mat shape: [dim, 3*hidden], split on axis=1
            h = mat.shape[1] // 3
            z, r, n = mat[:, :h], mat[:, h:2*h], mat[:, 2*h:]
            return np.concatenate([r, z, n], axis=1)

        def reorder_bias_vec(b: np.ndarray) -> np.ndarray:
            # b shape: [3*hidden]
            h = b.shape[0] // 3
            z, r, n = b[:h], b[h:2*h], b[2*h:]
            return np.concatenate([r, z, n], axis=0)

        # Forward direction: reorder and transpose to PyTorch shapes
        fw_ih_reordered = reorder_kernel(tf_fw_ih)
        fw_hh_reordered = reorder_kernel(tf_fw_hh)
        pytorch_fw_ih = torch.from_numpy(fw_ih_reordered.T).double()  # [3*hidden, input_size]
        pytorch_fw_hh = torch.from_numpy(fw_hh_reordered.T).double()  # [3*hidden, hidden]

        # Bias handling: support reset_after=True (shape [2, 3*hidden]) and reset_after=False (shape [3*hidden])
        if tf_fw_bias.ndim == 2 and tf_fw_bias.shape[0] == 2:
            # Keras reset_after=True: bias[0] is input bias, bias[1] is recurrent bias (z, r, h)
            fw_b_in = reorder_bias_vec(tf_fw_bias[0].copy())
            fw_b_rec = reorder_bias_vec(tf_fw_bias[1].copy())
            bias_mode = os.environ.get('ADTOF_GRU_BIAS_MODE', 'merge_candidate').lower()
            if bias_mode == 'separate_candidate':
                # Keep recurrent candidate bias separate to mirror Keras more directly
                pytorch_fw_bias_ih = torch.from_numpy(fw_b_in).double()
                pytorch_fw_bias_hh = torch.from_numpy(fw_b_rec).double()
            else:
                # Default: move candidate (n) recurrent bias into input bias to better match PyTorch equations
                h = fw_b_in.shape[0] // 3
                n_slice = slice(2*h, 3*h)
                fw_b_in[n_slice] += fw_b_rec[n_slice]
                fw_b_rec[n_slice] = 0.0
                pytorch_fw_bias_ih = torch.from_numpy(fw_b_in).double()
                pytorch_fw_bias_hh = torch.from_numpy(fw_b_rec).double()
        else:
            # Single bias vector case (reset_after=False)
            fw_b = reorder_bias_vec(tf_fw_bias.reshape(-1).copy())
            pytorch_fw_bias_ih = torch.from_numpy(fw_b).double()
            pytorch_fw_bias_hh = torch.zeros_like(pytorch_fw_bias_ih)

        # Backward direction: reorder and transpose
        bw_ih_reordered = reorder_kernel(tf_bw_ih)
        bw_hh_reordered = reorder_kernel(tf_bw_hh)
        pytorch_bw_ih = torch.from_numpy(bw_ih_reordered.T).double()
        pytorch_bw_hh = torch.from_numpy(bw_hh_reordered.T).double()

        if tf_bw_bias.ndim == 2 and tf_bw_bias.shape[0] == 2:
            bw_b_in = reorder_bias_vec(tf_bw_bias[0].copy())
            bw_b_rec = reorder_bias_vec(tf_bw_bias[1].copy())
            bias_mode = os.environ.get('ADTOF_GRU_BIAS_MODE', 'merge_candidate').lower()
            if bias_mode == 'separate_candidate':
                pytorch_bw_bias_ih = torch.from_numpy(bw_b_in).double()
                pytorch_bw_bias_hh = torch.from_numpy(bw_b_rec).double()
            else:
                h = bw_b_in.shape[0] // 3
                n_slice = slice(2*h, 3*h)
                bw_b_in[n_slice] += bw_b_rec[n_slice]
                bw_b_rec[n_slice] = 0.0
                pytorch_bw_bias_ih = torch.from_numpy(bw_b_in).double()
                pytorch_bw_bias_hh = torch.from_numpy(bw_b_rec).double()
        else:
            bw_b = reorder_bias_vec(tf_bw_bias.reshape(-1).copy())
            pytorch_bw_bias_ih = torch.from_numpy(bw_b).double()
            pytorch_bw_bias_hh = torch.zeros_like(pytorch_bw_bias_ih)
        
        print("  ✓ Reordered Keras gates (z,r,h) -> PyTorch (r,z,n); handled biases")
        
        # Store in PyTorch parameter names
        gru_prefix = f"gru_layers.{i}"
        
        gru_weights[f"{gru_prefix}.weight_ih_l0"] = pytorch_fw_ih
        gru_weights[f"{gru_prefix}.weight_hh_l0"] = pytorch_fw_hh  
        gru_weights[f"{gru_prefix}.bias_ih_l0"] = pytorch_fw_bias_ih
        gru_weights[f"{gru_prefix}.bias_hh_l0"] = pytorch_fw_bias_hh
        
        gru_weights[f"{gru_prefix}.weight_ih_l0_reverse"] = pytorch_bw_ih
        gru_weights[f"{gru_prefix}.weight_hh_l0_reverse"] = pytorch_bw_hh
        gru_weights[f"{gru_prefix}.bias_ih_l0_reverse"] = pytorch_bw_bias_ih  
        gru_weights[f"{gru_prefix}.bias_hh_l0_reverse"] = pytorch_bw_bias_hh
        
        print(f"  ✓ Converted GRU layer {i} weights successfully")
        print(f"    PyTorch shapes: ih={pytorch_fw_ih.shape}, hh={pytorch_fw_hh.shape}")
    
    print(f"\nTotal GRU weights converted: {len(gru_weights)}")
    return gru_weights


def extract_output_weights(tf_model, pytorch_model: ADTOFFrameRNN) -> Dict[str, torch.Tensor]:
    """
    Extract output layer weights from TensorFlow model.
    """
    print("\n=== Extracting Output Layer Weights ===")
    
    weights = tf_model.model.get_weights()
    
    # Output layer is typically the last dense layer
    output_weight = weights[-2]  # Weight matrix
    output_bias = weights[-1]    # Bias vector
    
    # Convert to PyTorch format (transpose weight matrix)
    pytorch_weight = torch.from_numpy(output_weight.T).double()
    pytorch_bias = torch.from_numpy(output_bias).double()
    
    print("Output layer:", output_weight.shape, "->", pytorch_weight.shape)
    
    return {
        "output_layer.weight": pytorch_weight,
        "output_layer.bias": pytorch_bias
    }


def convert_tensorflow_to_pytorch_weights(tf_model_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Convert all TensorFlow model weights to PyTorch format.
    
    Args:
        tf_model_path: Path to TensorFlow checkpoint (optional, uses default if None)
    
    Returns:
        Dictionary of PyTorch state dict
    """
    print("Loading TensorFlow model...")
    
    # Import here to avoid TF dependency if not needed
    from adtof.model.model import Model
    
    tf_model, hparams = Model.modelFactory(
        modelName="Frame_RNN",
        scenario="adtofAll", 
        fold=0,
        pre_trained_path=tf_model_path
    )
    
    if not tf_model.weightLoadedFlag:
        raise ValueError("TensorFlow model weights not loaded successfully")
    
    print("Creating corresponding PyTorch model...")
    # Use the correct n_bins that matches TensorFlow model
    from adtof.io.mir import getDim
    n_bins = getDim(**hparams)
    print(f"Using n_bins={n_bins} to match TensorFlow model")
    pytorch_model = create_frame_rnn_model(n_bins=n_bins)
    # Persist for save step metadata
    try:
        os.environ['ADTOF_N_BINS'] = str(int(n_bins))
    except Exception:
        pass
    
    # Analyze TF model structure
    analyze_tensorflow_model_structure(tf_model)
    
    # Extract weights by component
    all_weights = {}
    
    # CNN weights
    cnn_weights = extract_cnn_weights(tf_model, pytorch_model)
    all_weights.update(cnn_weights)
    
    # GRU weights (placeholder for now)
    gru_weights = extract_gru_weights(tf_model, pytorch_model)
    all_weights.update(gru_weights)
    
    # Output weights
    output_weights = extract_output_weights(tf_model, pytorch_model)
    all_weights.update(output_weights)
    
    print(f"\nTotal converted weights: {len(all_weights)}")
    
    return all_weights


def save_pytorch_weights(weights_dict: Dict[str, torch.Tensor], output_path: str):
    """
    Save PyTorch weights to file.
    """
    print("\nSaving PyTorch weights to:", output_path)
    
    # Add metadata
    # Try to infer n_bins from a representative tensor shape
    inferred_n_bins = None
    # Heuristic: output layer weight size gives final GRU size; n_bins not directly available.
    # We will persist n_bins if the conversion function stashed it in an env var.
    try:
        inferred_n_bins = int(os.environ.get('ADTOF_N_BINS', '0')) or None
    except Exception:
        inferred_n_bins = None

    weights_with_metadata = {
        'model_weights': weights_dict,
        'model_info': {
            'architecture': 'Frame_RNN',
            'n_bins': inferred_n_bins if inferred_n_bins is not None else 168,
            'conv_filters': [32, 64],
            'gru_units': [60, 60, 60],
            'output_classes': 5,
            'conversion_note': 'Converted from TensorFlow ADTOF model'
        }
    }
    
    torch.save(weights_with_metadata, output_path)
    print("Weights saved successfully!")
    
    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")


def load_pytorch_weights(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Load PyTorch weights from file.
    """
    print(f"Loading PyTorch weights from: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if 'model_weights' in checkpoint:
        weights = checkpoint['model_weights']
        info = checkpoint.get('model_info', {})
        print(f"Loaded weights for: {info.get('architecture', 'Unknown')}")
        print(f"Model info: {info}")
        return weights
    else:
        # Assume it's a direct state dict
        return checkpoint


def test_converted_weights():
    """
    Test the converted weights by loading them into a PyTorch model.
    """
    print("\n=== Testing Converted Weights ===")
    
    weights_path = "adtof_frame_rnn_pytorch_weights.pth"
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return
    
    # Load weights and model_info
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_weights' in checkpoint:
        weights = checkpoint['model_weights']
        info = checkpoint.get('model_info', {})
        n_bins = int(info.get('n_bins', 168))
    else:
        weights = checkpoint
        n_bins = 168
    # Create model with matching n_bins
    model = create_frame_rnn_model(n_bins=int(n_bins))
    model.eval()
    
    # Load available weights (skip missing ones like GRU)
    model_state = model.state_dict()
    loaded_keys = []
    missing_keys = []
    
    for key in model_state.keys():
        # Ignore PyTorch BatchNorm tracking buffers which don't exist in Keras
        if key.endswith('num_batches_tracked'):
            continue
        if key in weights:
            model_state[key] = weights[key]
            loaded_keys.append(key)
        else:
            missing_keys.append(key)
    
    model.load_state_dict(model_state)
    
    print(f"Loaded {len(loaded_keys)} weight tensors")
    print(f"Missing {len(missing_keys)} weight tensors: {missing_keys}")
    
    # Prepare identical random input for both models
    time_steps = 100
    test_input = torch.randn(1, time_steps, int(n_bins), 1)
    test_input_np = test_input.numpy().astype(np.float32)

    # PyTorch inference and intermediate checkpoints
    with torch.no_grad():
        # Full forward for end-to-end comparison
        output_pt = model(test_input).cpu().numpy()
        
        # Manual forward to capture intermediates aligned to Keras checkpoints
        x = test_input  # [B, T, F, C]
        B, T, FREQ, C = x.shape
        # CNN blocks operate on [B, C, T, F]
        x_nchw = x.permute(0, 3, 1, 2)
        # Stepwise capture for CNN parity
        def run_block_with_taps(block, xin):
            taps = {}
            z = xin
            for li, layer in enumerate(block):
                z = layer(z)
                # Map to Keras tap points (after conv act -> index 1 and 4; after BN -> 2 and 5; after pool -> 6)
                if li == 1:
                    taps['conv1_act'] = z.clone()
                elif li == 2:
                    taps['bn1'] = z.clone()
                elif li == 4:
                    taps['conv2_act'] = z.clone()
                elif li == 5:
                    taps['bn2'] = z.clone()
                elif li == 6:
                    taps['pool'] = z.clone()
            return z, taps

        pt_block0_out, pt_block0_taps = run_block_with_taps(model.cnn_blocks[0], x_nchw)
        pt_block1_out, pt_block1_taps = run_block_with_taps(model.cnn_blocks[1], pt_block0_out)
        pt_cnn0 = pt_block0_out
        pt_cnn1 = pt_block1_out
        # Keras 'sequential' output is channels-last [B, T, F, C]
        pt_seq_out = pt_cnn1.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        # Reshape to [B, T, features] before GRU (match Keras: channels-last before flatten)
        x_seq = pt_cnn1.permute(0, 2, 3, 1)  # [B, T, F, C]
        features = x_seq.shape[2] * x_seq.shape[3]  # F * C
        pt_flat = x_seq.reshape(B, T, features)
        # GRU layers
        pt_g0, _ = model.gru_layers[0](pt_flat)
        pt_g1, _ = model.gru_layers[1](pt_g0)
        pt_g2, _ = model.gru_layers[2](pt_g1)
        pt_logits = model.output_layer(pt_g2)
        pt_sigmoid = torch.sigmoid(pt_logits)
        
        # Convert to numpy for comparison
        pt_flat_np = pt_flat.cpu().numpy()
        pt_g0_np = pt_g0.cpu().numpy()
        pt_g1_np = pt_g1.cpu().numpy()
        pt_g2_np = pt_g2.cpu().numpy()
        pt_out_np = pt_sigmoid.cpu().numpy()
        # Also convert CNN taps to channels-last for comparison
        def nchw_to_nhwc(t):
            return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        pt_block0_taps_nhwc = {k: nchw_to_nhwc(v) for k, v in pt_block0_taps.items()}
        pt_block1_taps_nhwc = {k: nchw_to_nhwc(v) for k, v in pt_block1_taps.items()}

    # TensorFlow/Keras inference with original model (and intermediate checkpoints)
    from adtof.model.model import Model
    from adtof.io.mir import getDim
    import tensorflow as tf
    tf_model_path = None
    
    tf_model, hparams = Model.modelFactory(
        modelName="Frame_RNN",
        scenario="adtofAll", 
        fold=0,
        pre_trained_path=tf_model_path
    )
    
    if not tf_model.weightLoadedFlag:
        raise ValueError("TensorFlow model weights not loaded successfully in test")
    
    if not tf_model.weightLoadedFlag:
        print("Warning: TensorFlow model weights not loaded successfully; comparisons may be invalid")
    n_bins_tf = getDim(**hparams)
    if int(n_bins_tf) != int(n_bins):
        print(f"Warning: n_bins mismatch between TF ({n_bins_tf}) and PyTorch ({n_bins}); aborting comparison")
        return
    # Build intermediate Keras model by safe tensors derived from known layers
    tf_intermediates = {}
    try:
        available = {layer.name: layer for layer in tf_model.model.layers}
        # Expect these names from the functional builder
        reshape_layer = available.get('reshape', None)
        if reshape_layer is None:
            # Fallback: find by type
            reshape_candidates = [l for l in tf_model.model.layers if type(l).__name__ == 'Reshape']
            reshape_layer = reshape_candidates[0] if reshape_candidates else None
        # Tensors to fetch (global checkpoints)
        outputs = []
        names = []
        if reshape_layer is not None:
            # 'sequential' equivalent: tensor feeding reshape
            outputs.append(reshape_layer.input)
            names.append('sequential')
            # Flattened features
            outputs.append(reshape_layer.output)
            names.append('reshape')
        # Bidirectional layers in order
        bi_layers = [l for l in tf_model.model.layers if type(l).__name__ == 'Bidirectional']
        for idx, bl in enumerate(bi_layers):
            outputs.append(bl.output)
            names.append('bidirectional' if idx == 0 else f"bidirectional_{idx}")
        # Final dense output
        outputs.append(tf_model.model.output)
        names.append('denseOutput')
        tf_intermediate = tf.keras.Model(inputs=tf_model.model.input, outputs=outputs)
        with tf.device('/CPU:0'):
            tf_intermediates_list = tf_intermediate.predict(test_input_np, verbose=0)
        if not isinstance(tf_intermediates_list, list):
            tf_intermediates_list = [tf_intermediates_list]
        tf_intermediates = {name: arr for name, arr in zip(names, tf_intermediates_list)}
    except Exception as ie:
        print(f"  Skipping intermediate layer comparisons due to error: {ie}")

    # CNN stepwise taps using the nested sequential layer
    tf_cnn_taps = {}
    try:
        # Find the Sequential CNN layer robustly (name can be 'sequential' or 'sequential_X')
        seq_layer = None
        for l in tf_model.model.layers:
            if getattr(l, 'name', '').startswith('sequential') or type(l).__name__ == 'Sequential':
                seq_layer = l
                break
        if seq_layer is None:
            raise ValueError("Sequential CNN layer not found")
        # Reconstruct the sequential forward to capture inner taps using the same weights
        tf_in = tf.keras.Input(shape=(None, int(n_bins), 1), dtype=tf.float32)
        z = tf_in
        sublayers = list(seq_layer.layers)
        def add(name, tensor):
            tf_cnn_taps[name] = tensor
        # Expect blocks of 6 layers: conv, bn, conv, bn, pool, drop
        block_idx = 0
        i = 0
        while i + 5 < len(sublayers):
            conv1, bn1, conv2, bn2, pool, drop = sublayers[i:i+6]
            z = conv1(z); add(f'block{block_idx}_conv1_act', z)
            z = bn1(z);   add(f'block{block_idx}_bn1', z)
            z = conv2(z); add(f'block{block_idx}_conv2_act', z)
            z = bn2(z);   add(f'block{block_idx}_bn2', z)
            z = pool(z);  add(f'block{block_idx}_pool', z)
            z = drop(z)
            block_idx += 1
            i += 6
        tf_cnn_model = tf.keras.Model(tf_in, list(tf_cnn_taps.values()))
        with tf.device('/CPU:0'):
            tf_cnn_list = tf_cnn_model.predict(test_input_np, verbose=0)
        tf_cnn_taps = {name: arr for name, arr in zip(tf_cnn_taps.keys(), tf_cnn_list)}
    except Exception as e:
        print(f"  Skipping CNN step taps due to error: {e}")
    
    # Final TF output (CPU)
    with tf.device('/CPU:0'):
        output_tf = tf_model.model.predict(test_input_np, verbose=0)

    # Compare outputs
    if output_tf.shape != output_pt.shape:
        print(f"Warning: Output shape mismatch TF {output_tf.shape} vs PT {output_pt.shape}")
    min_shape_time = min(output_tf.shape[1], output_pt.shape[1])
    min_shape_classes = min(output_tf.shape[2], output_pt.shape[2])
    tf_cmp = output_tf[:, :min_shape_time, :min_shape_classes]
    pt_cmp = output_pt[:, :min_shape_time, :min_shape_classes]
    mae = np.mean(np.abs(tf_cmp - pt_cmp))
    max_abs = np.max(np.abs(tf_cmp - pt_cmp))
    mse = np.mean((tf_cmp - pt_cmp) ** 2)
    print("Comparison with original Keras model:")
    print(f"  Shapes TF {output_tf.shape} vs PT {output_pt.shape}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max |diff|: {max_abs:.6f}")

    # Layer-by-layer diagnostics
    def _report(name: str, a: np.ndarray, b: np.ndarray):
        if a is None or b is None:
            print(f"  {name}: missing tensors for comparison")
            return
        if a.shape != b.shape:
            min_t = min(a.shape[1], b.shape[1]) if a.ndim > 1 else None
            min_c = min(a.shape[-1], b.shape[-1]) if a.ndim > 1 else None
            a_c = a
            b_c = b
            if a.ndim == 4 and b.ndim == 4:
                a_c = a[:, :min_t, :min_c, :]
                b_c = b[:, :min_t, :min_c, :]
            elif a.ndim == 3 and b.ndim == 3:
                min_t = min(a.shape[1], b.shape[1])
                min_f = min(a.shape[2], b.shape[2])
                a_c = a[:, :min_t, :min_f]
                b_c = b[:, :min_t, :min_f]
            elif a.ndim == 2 and b.ndim == 2:
                min_f = min(a.shape[1], b.shape[1])
                a_c = a[:, :min_f]
                b_c = b[:, :min_f]
        else:
            a_c = a
            b_c = b
        mae_l = float(np.mean(np.abs(a_c - b_c)))
        mse_l = float(np.mean((a_c - b_c) ** 2))
        max_l = float(np.max(np.abs(a_c - b_c)))
        print(f"  {name}: MAE={mae_l:.6f} MSE={mse_l:.6f} MAX={max_l:.6f} shape_tf={a.shape} shape_pt={b.shape}")

    print("\nLayer-by-layer comparison:")
    # After CNN sequential (channels-last)
    tf_seq = tf_intermediates.get('sequential')
    _report('sequential', tf_seq, pt_seq_out)
    # After reshape to [B, T, features]
    tf_flat = tf_intermediates.get('reshape')
    _report('reshape', tf_flat, pt_flat_np)
    # GRU layers (bidirectional outputs)
    tf_g0 = tf_intermediates.get('bidirectional')
    _report('bidirectional_0', tf_g0, pt_g0_np)
    tf_g1 = tf_intermediates.get('bidirectional_1')
    _report('bidirectional_1', tf_g1, pt_g1_np)
    tf_g2 = tf_intermediates.get('bidirectional_2')
    _report('bidirectional_2', tf_g2, pt_g2_np)
    # Final output (post-sigmoid)
    tf_out = tf_intermediates.get('denseOutput', output_tf)
    _report('output', tf_out, pt_out_np)

    # Detailed CNN step comparisons per block
    if tf_cnn_taps:
        print("\nCNN detailed comparisons:")
        # Block 0
        def rep(name, tf_arr, pt_arr):
            _report(name, tf_arr, pt_arr)
        rep('block0_conv1_act', tf_cnn_taps.get('block0_conv1_act'), pt_block0_taps_nhwc.get('conv1_act'))
        rep('block0_bn1', tf_cnn_taps.get('block0_bn1'), pt_block0_taps_nhwc.get('bn1'))
        rep('block0_conv2_act', tf_cnn_taps.get('block0_conv2_act'), pt_block0_taps_nhwc.get('conv2_act'))
        rep('block0_bn2', tf_cnn_taps.get('block0_bn2'), pt_block0_taps_nhwc.get('bn2'))
        rep('block0_pool', tf_cnn_taps.get('block0_pool'), pt_block0_taps_nhwc.get('pool'))
        # Block 1
        rep('block1_conv1_act', tf_cnn_taps.get('block1_conv1_act'), pt_block1_taps_nhwc.get('conv1_act'))
        rep('block1_bn1', tf_cnn_taps.get('block1_bn1'), pt_block1_taps_nhwc.get('bn1'))
        rep('block1_conv2_act', tf_cnn_taps.get('block1_conv2_act'), pt_block1_taps_nhwc.get('conv2_act'))
        rep('block1_bn2', tf_cnn_taps.get('block1_bn2'), pt_block1_taps_nhwc.get('bn2'))
        rep('block1_pool', tf_cnn_taps.get('block1_pool'), pt_block1_taps_nhwc.get('pool'))


if __name__ == "__main__":
    try:
        print("ADTOF TensorFlow to PyTorch Weight Converter")
        print("=" * 50)
        
        # Convert weights
        weights = convert_tensorflow_to_pytorch_weights()
        
        # Save to file
        output_path = "adtof_frame_rnn_pytorch_weights.pth"
        save_pytorch_weights(weights, output_path)
        
        # Test the converted weights
        test_converted_weights()
        
        print("\n" + "=" * 50)
        print("Conversion completed successfully!")
        print(f"PyTorch weights saved to: {output_path}")
        print("\nTo use the converted model:")
        print("  from adtof_pytorch import create_frame_rnn_model")
        print("  model = create_frame_rnn_model()")
        print(f"  model.load_state_dict(torch.load('{output_path}')['model_weights'], strict=False)")
        
    except Exception as e:
        print(f"\nConversion failed: {e}")
        print("Make sure TensorFlow model is available and properly configured.")
