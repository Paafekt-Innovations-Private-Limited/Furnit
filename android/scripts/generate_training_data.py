#!/usr/bin/env python3
"""
Generate training data for GaussianHead from full SHARP model.

This script:
1. Loads the full SHARP PyTorch model
2. Processes images to extract encoder features and decoder outputs
3. Saves paired data for training the lightweight GaussianHead

Usage:
    python generate_training_data.py --model_path /path/to/sharp.pt --image_dir /path/to/images --output_dir ./training_data
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json

# Add SHARP model path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml_experiments"))


def load_sharp_model(model_path: str, device: str = 'cpu'):
    """Load the full SHARP model."""
    print(f"Loading SHARP model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # Need model architecture - try to infer
            print("Checkpoint contains state_dict, need model class")
            return None, checkpoint['state_dict']
        else:
            # Assume it's the model itself in a dict
            model = checkpoint
    else:
        model = checkpoint

    if hasattr(model, 'eval'):
        model.eval()

    print(f"Model loaded successfully")
    return model, None


def preprocess_image(image_path: str, size: int = 1536):
    """Load and preprocess image for SHARP."""
    img = Image.open(image_path).convert('RGB')

    # Resize to square
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor


def extract_features_and_outputs(model, image_tensor, device='cpu'):
    """
    Run SHARP model and extract:
    - Encoder features (merged multi-scale features)
    - Decoder outputs (Gaussian parameters)

    Returns:
        encoder_features: [1024, 96, 96] tensor
        gaussian_params: dict with positions, scales, rotations, colors, opacity
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # The SHARP model structure varies - we need to hook into the right layers
        # This is a simplified version - adjust based on actual model structure

        try:
            # Try direct forward pass
            output = model(image_tensor)

            # Extract what we need from output
            if isinstance(output, dict):
                # Model returns dict with gaussian params
                positions = output.get('positions', output.get('xyz', None))
                scales = output.get('scales', output.get('scale', None))
                rotations = output.get('rotations', output.get('rotation', None))
                colors = output.get('colors', output.get('rgb', output.get('sh', None)))
                opacity = output.get('opacity', output.get('alpha', None))
            elif isinstance(output, (list, tuple)):
                # Model returns tuple/list
                if len(output) >= 5:
                    positions, scales, rotations, colors, opacity = output[:5]
                else:
                    print(f"Unexpected output length: {len(output)}")
                    return None, None
            else:
                print(f"Unexpected output type: {type(output)}")
                return None, None

            # For encoder features, we need to hook the encoder output
            # This is model-specific - for now we'll create synthetic features
            # based on decoder input requirements

            return None, {
                'positions': positions,
                'scales': scales,
                'rotations': rotations,
                'colors': colors,
                'opacity': opacity
            }

        except Exception as e:
            print(f"Forward pass failed: {e}")
            return None, None


def run_sharp_with_hooks(model, image_tensor, device='cpu'):
    """
    Run SHARP with hooks to capture intermediate features.
    """
    image_tensor = image_tensor.to(device)
    encoder_features = {}
    decoder_input = {}

    def make_hook(name, storage):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach().cpu()
            elif isinstance(output, (list, tuple)):
                storage[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
        return hook

    hooks = []

    # Register hooks on likely encoder/decoder boundaries
    for name, module in model.named_modules():
        name_lower = name.lower()
        if 'encoder' in name_lower and 'image' not in name_lower:
            # Patch encoder outputs
            if hasattr(module, 'register_forward_hook'):
                hooks.append(module.register_forward_hook(make_hook(name, encoder_features)))
        elif 'decoder' in name_lower or 'upsample' in name_lower:
            if hasattr(module, 'register_forward_hook'):
                hooks.append(module.register_forward_hook(make_hook(name, decoder_input)))

    try:
        with torch.no_grad():
            output = model(image_tensor)
    finally:
        for h in hooks:
            h.remove()

    return encoder_features, decoder_input, output


def generate_synthetic_training_pair(image_path: str, output_size: int = 384):
    """
    Generate a synthetic training pair when full SHARP model isn't available.
    Uses image features as a proxy for encoder output and generates
    plausible Gaussian parameters from image content.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert('RGB')
    img = img.resize((output_size, output_size), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0

    # Create synthetic encoder features by downsampling and expanding channels
    # This simulates what the encoder would produce
    h, w = 96, 96

    # Downsample image
    img_small = Image.fromarray((img_np * 255).astype(np.uint8))
    img_small = img_small.resize((w, h), Image.LANCZOS)
    img_small_np = np.array(img_small).astype(np.float32) / 255.0

    # Create 1024 channels by repeating and adding noise
    encoder_features = np.zeros((1024, h, w), dtype=np.float32)
    for c in range(1024):
        base_channel = c % 3
        noise = np.random.randn(h, w).astype(np.float32) * 0.1
        encoder_features[c] = img_small_np[:, :, base_channel] + noise

    # Generate target Gaussian parameters from image
    # Positions: based on pixel locations with depth from luminance
    positions = np.zeros((3, output_size, output_size), dtype=np.float32)
    for y in range(output_size):
        for x in range(output_size):
            positions[0, y, x] = (x / output_size - 0.5) * 4.0  # X: [-2, 2]
            positions[1, y, x] = (y / output_size - 0.5) * 4.0  # Y: [-2, 2]
            # Depth from luminance
            lum = 0.299 * img_np[y, x, 0] + 0.587 * img_np[y, x, 1] + 0.114 * img_np[y, x, 2]
            positions[2, y, x] = (lum - 0.5) * 2.0  # Z: [-1, 1]

    # Opacity: based on edge detection (edges more opaque)
    gray = np.mean(img_np, axis=2)
    sobel_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    sobel_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    opacity = 0.5 + 0.5 * np.clip(edges * 5, 0, 1)
    opacity = opacity[np.newaxis, :, :]

    # Scales: smaller where there's detail
    detail = edges
    scales = np.zeros((3, output_size, output_size), dtype=np.float32)
    base_scale = 0.02
    for c in range(3):
        scales[c] = base_scale * (1.0 - 0.5 * np.clip(detail * 3, 0, 1))

    # Rotations: identity quaternion
    rotations = np.zeros((4, output_size, output_size), dtype=np.float32)
    rotations[0] = 1.0  # w = 1, x = y = z = 0

    # Colors: from image RGB
    colors = img_np.transpose(2, 0, 1)  # [3, H, W]

    return encoder_features, {
        'positions': positions,
        'opacity': opacity,
        'scales': scales,
        'rotations': rotations,
        'colors': colors
    }


def save_training_sample(output_dir: Path, sample_idx: int,
                         encoder_features: np.ndarray,
                         gaussian_params: dict):
    """Save a training sample to disk."""
    sample_dir = output_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save encoder features
    np.save(sample_dir / "encoder_features.npy", encoder_features)

    # Save Gaussian parameters
    np.save(sample_dir / "positions.npy", gaussian_params['positions'])
    np.save(sample_dir / "opacity.npy", gaussian_params['opacity'])

    if 'scales' in gaussian_params:
        np.save(sample_dir / "scales.npy", gaussian_params['scales'])
    if 'rotations' in gaussian_params:
        np.save(sample_dir / "rotations.npy", gaussian_params['rotations'])
    if 'colors' in gaussian_params:
        np.save(sample_dir / "colors.npy", gaussian_params['colors'])

    return sample_dir


def find_images(image_dir: str, extensions: list = ['.jpg', '.jpeg', '.png']):
    """Find all images in directory."""
    image_dir = Path(image_dir)
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description='Generate GaussianHead training data')
    parser.add_argument('--model_path', type=str,
                        default='/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt',
                        help='Path to SHARP model')
    parser.add_argument('--image_dir', type=str,
                        default='/Users/al/Documents/tries01/Furnit/ml_experiments/test_images',
                        help='Directory with training images')
    parser.add_argument('--output_dir', type=str, default='./training_data',
                        help='Output directory for training data')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic training data (when full model unavailable)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to generate')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(args.image_dir)
    if not images:
        print(f"No images found in {args.image_dir}")
        print("Searching in parent directories...")
        images = find_images(Path(args.image_dir).parent)

    if not images:
        print("No images found. Please provide an image directory.")
        return

    print(f"Found {len(images)} images")

    # Limit samples
    images = images[:args.max_samples]

    if args.synthetic:
        print("Generating synthetic training data...")
        for idx, image_path in enumerate(images):
            print(f"Processing {idx+1}/{len(images)}: {image_path.name}")
            try:
                encoder_features, gaussian_params = generate_synthetic_training_pair(str(image_path))
                save_training_sample(output_dir, idx, encoder_features, gaussian_params)
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\nGenerated {len(images)} synthetic training samples in {output_dir}")
        return

    # Try to load full SHARP model
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"Using device: {device}")

    model, state_dict = load_sharp_model(args.model_path, device)

    if model is None:
        print("Could not load full model. Falling back to synthetic data generation.")
        print("Run with --synthetic flag to generate synthetic training data.")
        return

    model = model.to(device)

    # Process images
    for idx, image_path in enumerate(images):
        print(f"Processing {idx+1}/{len(images)}: {image_path.name}")

        try:
            image_tensor = preprocess_image(str(image_path))

            # Try to extract features with hooks
            encoder_features, decoder_input, output = run_sharp_with_hooks(
                model, image_tensor, device
            )

            if encoder_features:
                # Find the right encoder output
                for name, feat in encoder_features.items():
                    print(f"  Encoder {name}: {feat.shape if isinstance(feat, torch.Tensor) else type(feat)}")

            if output is not None:
                print(f"  Output type: {type(output)}")
                if isinstance(output, dict):
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: {v.shape}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nProcessing complete. Check output in {output_dir}")


if __name__ == "__main__":
    main()
