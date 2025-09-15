"""
Example usage of the PyTorch ADTOF Frame_RNN model.

This demonstrates how to:
1. Load the PyTorch model
2. Load converted weights
3. Run inference on audio data
4. Interpret the results
"""

import torch
import numpy as np
import os
from adtof_pytorch import create_frame_rnn_model, load_pytorch_weights, load_audio_for_model, calculate_n_bins, transcribe_to_midi
from adtof.io.mir import preProcess

def main():
    print("ADTOF PyTorch Model Usage Example")
    print("=" * 40)
    
    # 1. Create the PyTorch model
    print("1. Creating PyTorch model...")
    n_bins = calculate_n_bins()  # Calculate correct number of bins
    model = create_frame_rnn_model(n_bins=n_bins)
    print(f"   Model created with {model.get_model_info()['total_parameters']:,} parameters")
    print(f"   Using {n_bins} frequency bins")
    
    # 2. Load converted weights
    weights_path = "adtof_frame_rnn_pytorch_weights.pth"
    print(f"\n2. Loading converted weights from {weights_path}...")
    
    try:
        model = load_pytorch_weights(model, weights_path, strict=False)
        print("   ✓ Weights loaded successfully!")
        weights_loaded = True
    except FileNotFoundError:
        print("   ⚠ Weights file not found. Using random initialization.")
        print("   Run 'python convert_weights.py' to convert TensorFlow weights.")
        weights_loaded = False
    
    # 3. Prepare model for inference
    model.eval()  # Set to evaluation mode
    
    # 4. Load real audio file
    print("\n3. Loading audio file...")
    audio_path = "test.wav"
    
    if os.path.exists(audio_path):
        print(f"   Loading: {audio_path}")
        try:
            audio_features = load_audio_for_model(audio_path)
            print(f"   ✓ Audio loaded successfully!")
            print(f"   Input shape: {audio_features.shape}")
            print(f"   Duration: {audio_features.shape[1] / 100:.1f} seconds")
            using_real_audio = True
        except Exception as e:
            print(f"   ✗ Failed to load audio: {e}")
            print("   Using dummy data instead...")
            using_real_audio = False
    else:
        print(f"   Audio file not found: {audio_path}")
        print("   Using dummy data instead...")
        using_real_audio = False
    
    if not using_real_audio:
        # Fallback to dummy data
        batch_size = 1
        time_steps = 400  # ~4 seconds at 100 FPS
        audio_features = torch.randn(batch_size, time_steps, n_bins, 1)
        print(f"   Dummy input shape: {audio_features.shape}")
        print("   (Replace with real audio file for actual transcription)")
    
    # 5. Run inference
    print("\n4. Running inference...")
    with torch.no_grad():
        predictions = model(audio_features)
    
    print(f"   Output shape: {predictions.shape}")
    print(f"   Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    # 6. Interpret results
    print("\n5. Interpreting results...")
    
    # Class names for the 5 drum instruments
    class_names = ["Bass Drum (BD)", "Snare Drum (SD)", "Hi-Hat (HH)", "Tom-Tom (TT)", "Cymbal (CY)"]
    
    # Apply threshold to get binary predictions
    threshold = 0.5
    binary_predictions = (predictions > threshold).float()
    
    # Count detections per class
    detections_per_class = binary_predictions.sum(dim=1).squeeze()  # Sum over time dimension
    
    print("   Drum detections (with threshold=0.5):")
    for i, class_name in enumerate(class_names):
        count = int(detections_per_class[i].item())
        print(f"     {class_name}: {count} hits")
    
    # Show some example time steps with high confidence
    print(f"\n   Example high-confidence predictions (>{threshold}):")
    high_conf_mask = predictions.squeeze() > threshold
    
    for t in range(min(10, predictions.shape[1])):  # Show first 10 time steps
        frame_preds = predictions[0, t, :]
        if torch.any(frame_preds > threshold):
            active_classes = [class_names[i] for i in range(5) if frame_preds[i] > threshold]
            max_conf = torch.max(frame_preds).item()
            print(f"     Time {t:3d}: {', '.join(active_classes)} (max conf: {max_conf:.3f})")
    
    # 7. Save predictions (optional)
    print(f"\n6. Saving predictions...")
    output_path = "example_predictions.npy"
    np.save(output_path, predictions.numpy())
    print(f"   Predictions saved to: {output_path}")
    
    print(f"\n✓ Example completed successfully!")
    
    if not weights_loaded:
        print("\nNote: This example used random weights.")
        print("For real drum transcription, convert the TensorFlow weights first:")
        print("  python convert_weights.py")


def load_and_use_model(weights_path: str = "adtof_frame_rnn_pytorch_weights.pth"):
    """
    Convenience function to load the model with weights.
    
    Returns:
        Loaded PyTorch model ready for inference
    """
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins=n_bins)
    model = load_pytorch_weights(model, weights_path, strict=False)
    model.eval()
    return model


def load_audio_for_model_madmom(audio_path: str):
    audio_tensor = preProcess(audio_path)
    audio_tensor = torch.from_numpy(audio_tensor).float().unsqueeze(0)
    return audio_tensor

def transcribe_audio_file(audio_path: str, weights_path: str = "adtof_frame_rnn_pytorch_weights.pth", threshold=None):
    """
    Transcribe a single audio file to drum events.
    
    Args:
        audio_path: Path to audio file
        weights_path: Path to PyTorch weights
        threshold: Detection threshold (float or list of 5 floats for each class)
                  If None, uses ADTOF Frame_RNN optimized thresholds
        
    Returns:
        Dictionary with drum events per class
    """
    print(f"Transcribing: {audio_path}")
    
    # Load model
    model = load_and_use_model(weights_path)
    
    # Load and process audio
    # audio_tensor = load_audio_for_model(audio_path)
    audio_tensor = load_audio_for_model_madmom(audio_path)
    
    # Run inference
    with torch.no_grad():
        predictions = model(audio_tensor)
    
    # Extract drum events
    class_names = ["Bass Drum (BD)", "Snare Drum (SD)", "Hi-Hat (HH)", "Tom-Tom (TT)", "Cymbal (CY)"]
    
    # Use ADTOF Frame_RNN optimized thresholds if none provided
    if threshold is None:
        # From ADTOF hyperparameters: [BD, SD, TT, HH, CY]
        adtof_thresholds = [0.22, 0.24, 0.32, 0.22, 0.30]
        print(f"Using ADTOF optimized thresholds: {adtof_thresholds}")
    elif isinstance(threshold, (int, float)):
        adtof_thresholds = [threshold] * 5
        print(f"Using uniform threshold: {threshold}")
    else:
        adtof_thresholds = threshold
        print(f"Using custom thresholds: {adtof_thresholds}")
    
    drum_events = {}
    
    # Apply per-class thresholds
    predictions_np = predictions.squeeze().numpy()  # [time, classes]
    
    for class_idx, class_name in enumerate(class_names):
        # Apply class-specific threshold
        class_threshold = adtof_thresholds[class_idx]
        class_preds = predictions_np[:, class_idx] > class_threshold
        onsets = []
        
        for t in range(1, len(class_preds)):
            if class_preds[t] and not class_preds[t-1]:  # Onset detected
                time_seconds = t / 100.0  # Convert frame to seconds (100 FPS)
                confidence = predictions_np[t, class_idx]
                onsets.append((time_seconds, confidence))
        
        drum_events[class_name] = onsets
    
    return drum_events


if __name__ == "__main__":
    out = transcribe_to_midi("test.wav", "test.mid")
    print(f"Wrote {out}")
