#!/usr/bin/env python3
"""
Visualize ADTOF PyTorch model activations for debugging.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from adtof_pytorch import create_frame_rnn_model, load_pytorch_weights, load_audio_for_model
import os


def visualize_model_activations(audio_path: str = "test.wav", 
                              weights_path: str = "adtof_frame_rnn_pytorch_weights.pth",
                              output_path: str = "model_activations.png"):
    print("ADTOF PyTorch Model Activation Visualization")
    print("=" * 50)
    print(f"Loading model and weights from: {weights_path}")
    model = create_frame_rnn_model(n_bins=84)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_weights' in checkpoint:
        model.load_state_dict(checkpoint['model_weights'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print("✓ Model loaded successfully")

    print(f"Loading audio: {audio_path}")
    audio_tensor = load_audio_for_model(audio_path)
    audio_tensor = audio_tensor.to(device)
    print(f"Audio tensor shape: {audio_tensor.shape}")

    print("Running inference...")
    with torch.no_grad():
        predictions = model(audio_tensor)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")

    predictions_np = predictions.squeeze(0).cpu().numpy()

    class_names = ["Bass Drum (BD)", "Snare Drum (SD)", "Hi-Hat (HH)", "Tom-Tom (TT)", "Cymbal (CY)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'ADTOF PyTorch Model Activations - {os.path.basename(audio_path)}', fontsize=16)

    for i in range(5):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        class_activations = predictions_np[:, i]
        ax.plot(class_activations, linewidth=1)
        ax.set_title(f'{class_names[i]}\nMax: {class_activations.max():.4f}')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Activation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        adtof_thresholds = [0.22, 0.24, 0.32, 0.22, 0.30]
        ax.axhline(y=adtof_thresholds[i], color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold: {adtof_thresholds[i]}')
        ax.legend()

    ax = axes[1, 2]
    im = ax.imshow(predictions_np.T, aspect='auto', origin='lower', 
                   cmap='viridis', vmin=0, vmax=1, interpolation='none')
    ax.set_title('All Class Activations (Heatmap)')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Drum Class')
    ax.set_yticks(range(5))
    ax.set_yticklabels([name.split('(')[1].rstrip(')') for name in class_names])

    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    return predictions_np


if __name__ == "__main__":
    if not os.path.exists("test.wav"):
        print("Error: test.wav not found!")
        exit(1)
    visualize_model_activations()
    print("\n" + "=" * 50)
    print("Visualization complete! Check 'model_activations.png'")
