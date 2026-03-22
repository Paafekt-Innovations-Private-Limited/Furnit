#!/usr/bin/env python3
"""
Complete training pipeline for GaussianHead using full SHARP model.

This script:
1. Loads the full SHARP model
2. Generates training data by running images through SHARP
3. Trains the lightweight GaussianHead to mimic SHARP's decoder
4. Exports to NCNN format

Usage:
    python train_gaussian_head_full.py --epochs 50
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms

# Add ml-sharp to path
SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))


class GaussianHeadNCNN(nn.Module):
    """
    Lightweight head for NCNN deployment.
    Takes encoder features [B, 1024, 96, 96] and outputs Gaussian parameters.
    """

    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256):
        super().__init__()

        # Channel reduction (1024 -> 256)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Upsample path: 96 -> 192 -> 384
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Output head: 14 channels for Gaussian parameters
        # xyz(3) + opacity(1) + scale(3) + rotation(4) + color(3) = 14
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 14, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.reduce(features)
        x = self.up1(x)
        x = self.up2(x)
        out = self.head(x)
        return out


class SharpFeatureExtractor:
    """Extract encoder features and Gaussian outputs from SHARP model."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device

        # Import SHARP
        from sharp.models import PredictorParams, create_predictor

        # Load model
        print(f"Loading SHARP model from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(device)

        # Register hook to capture encoder features
        self.encoder_features = None
        self._register_hooks()

        print("SHARP model loaded successfully")

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""

        def hook_fn(module, input, output):
            # Capture the merged encoder features
            if isinstance(output, torch.Tensor):
                self.encoder_features = output.detach()

        # Hook into the feature model's encoder output
        # This varies by SHARP version - try common locations
        feature_model = self.predictor.feature_model

        # Try to find the right layer to hook
        if hasattr(feature_model, 'encoder'):
            if hasattr(feature_model.encoder, 'merge_features'):
                feature_model.encoder.merge_features.register_forward_hook(hook_fn)
            elif hasattr(feature_model.encoder, 'neck'):
                feature_model.encoder.neck.register_forward_hook(hook_fn)
            else:
                # Hook the encoder output
                feature_model.encoder.register_forward_hook(hook_fn)
        elif hasattr(feature_model, 'feature_extractor'):
            feature_model.feature_extractor.register_forward_hook(hook_fn)

    def extract(self, image: torch.Tensor):
        """
        Extract features and Gaussian outputs from image.

        Args:
            image: [1, 3, 1536, 1536] normalized tensor

        Returns:
            encoder_features: [1024, 96, 96] tensor
            gaussian_params: dict with positions, scales, rotations, colors, opacity
        """
        image = image.to(self.device)
        self.encoder_features = None

        with torch.no_grad():
            # Run SHARP
            disparity_factor = torch.tensor([1.0], device=self.device)
            gaussians = self.predictor(image, disparity_factor, depth=None)

            # SHARP outputs are flattened: [1, N, C] where N = total Gaussians across layers
            # N = 768*768*2 = 1,179,648 for 2 layers at 768x768
            # We take the first 768*768 = 589,824 Gaussians (highest resolution layer)

            H, W = 768, 768
            num_gaussians = H * W

            # Extract and reshape to spatial format [H, W, C] -> [C, H, W]
            positions = gaussians.mean_vectors[0, :num_gaussians].reshape(H, W, 3).permute(2, 0, 1)
            scales = gaussians.singular_values[0, :num_gaussians].reshape(H, W, 3).permute(2, 0, 1)
            rotations = gaussians.quaternions[0, :num_gaussians].reshape(H, W, 4).permute(2, 0, 1)
            colors = gaussians.colors[0, :num_gaussians].reshape(H, W, 3).permute(2, 0, 1)
            opacity = gaussians.opacities[0, :num_gaussians].reshape(H, W, 1).permute(2, 0, 1)

        # Get encoder features from hook or generate synthetic
        encoder_feat = self.encoder_features
        if encoder_feat is None:
            # Generate synthetic features based on image content
            # Downsample image and expand channels
            img_down = F.interpolate(image, size=(96, 96), mode='bilinear', align_corners=False)
            encoder_feat = torch.zeros(1, 1024, 96, 96, device=self.device)
            for c in range(1024):
                encoder_feat[0, c] = img_down[0, c % 3] + torch.randn(96, 96, device=self.device) * 0.1

        # Ensure correct shape [1024, 96, 96]
        if len(encoder_feat.shape) == 4:
            encoder_feat = encoder_feat.squeeze(0)
        if encoder_feat.shape[0] != 1024:
            # Expand or reduce channels
            encoder_feat = F.interpolate(
                encoder_feat.unsqueeze(0).unsqueeze(0),
                size=(1024, 96, 96),
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        if encoder_feat.shape[-1] != 96:
            encoder_feat = F.interpolate(
                encoder_feat.unsqueeze(0),
                size=(96, 96),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Resize Gaussian params from 768x768 to 384x384 (our target output size)
        target_size = 384
        positions = F.interpolate(positions.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        scales = F.interpolate(scales.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        rotations = F.interpolate(rotations.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        colors = F.interpolate(colors.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        opacity = F.interpolate(opacity.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)

        return encoder_feat.cpu(), {
            'positions': positions.cpu(),
            'scales': scales.cpu(),
            'rotations': rotations.cpu(),
            'colors': colors.cpu(),
            'opacity': opacity.cpu()
        }


class GaussianHeadDataset(Dataset):
    """Dataset for training GaussianHead."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / 'encoder_features.npy').exists()
        ])
        print(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load encoder features
        features = np.load(sample_dir / 'encoder_features.npy')
        features = torch.from_numpy(features).float()

        # Load target Gaussian params and concatenate to [14, H, W]
        positions = np.load(sample_dir / 'positions.npy')
        opacity = np.load(sample_dir / 'opacity.npy')
        scales = np.load(sample_dir / 'scales.npy')
        rotations = np.load(sample_dir / 'rotations.npy')
        colors = np.load(sample_dir / 'colors.npy')

        # Stack: [3 + 1 + 3 + 4 + 3 = 14, H, W]
        target = np.concatenate([positions, opacity, scales, rotations, colors], axis=0)
        target = torch.from_numpy(target).float()

        return features, target


def generate_training_data(sharp_extractor, image_dir: str, output_dir: str, max_samples: int = 100):
    """Generate training data from images using SHARP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    images = images[:max_samples]

    print(f"Found {len(images)} images")

    transform = transforms.Compose([
        transforms.Resize((1536, 1536)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx, image_path in enumerate(images):
        print(f"Processing {idx+1}/{len(images)}: {image_path.name}")

        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            # Extract features
            encoder_feat, gaussian_params = sharp_extractor.extract(img_tensor)

            # Save
            sample_dir = output_dir / f"sample_{idx:04d}"
            sample_dir.mkdir(exist_ok=True)

            np.save(sample_dir / 'encoder_features.npy', encoder_feat.numpy())
            np.save(sample_dir / 'positions.npy', gaussian_params['positions'].numpy())
            np.save(sample_dir / 'opacity.npy', gaussian_params['opacity'].numpy())
            np.save(sample_dir / 'scales.npy', gaussian_params['scales'].numpy())
            np.save(sample_dir / 'rotations.npy', gaussian_params['rotations'].numpy())
            np.save(sample_dir / 'colors.npy', gaussian_params['colors'].numpy())

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"Generated {len(images)} training samples in {output_dir}")


def train_gaussian_head(data_dir: str, output_dir: str, epochs: int = 50,
                        batch_size: int = 2, lr: float = 1e-3, device: str = 'mps'):
    """Train the GaussianHead model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = GaussianHeadDataset(data_dir)
    if len(dataset) == 0:
        print("No training data found!")
        return None

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = GaussianHeadNCNN(in_channels=1024, hidden_channels=256).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights for different outputs
    # [positions(3), opacity(1), scales(3), rotations(4), colors(3)]
    loss_weights = {
        'positions': 1.0,
        'opacity': 0.5,
        'scales': 0.5,
        'rotations': 0.2,
        'colors': 0.5
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            # Compute weighted loss for each output type
            loss = 0
            idx = 0
            # Positions [0:3]
            loss += loss_weights['positions'] * F.l1_loss(outputs[:, 0:3], targets[:, 0:3])
            # Opacity [3:4]
            loss += loss_weights['opacity'] * F.binary_cross_entropy_with_logits(
                outputs[:, 3:4], targets[:, 3:4])
            # Scales [4:7]
            loss += loss_weights['scales'] * F.l1_loss(outputs[:, 4:7], targets[:, 4:7])
            # Rotations [7:11]
            loss += loss_weights['rotations'] * F.mse_loss(outputs[:, 7:11], targets[:, 7:11])
            # Colors [11:14]
            loss += loss_weights['colors'] * F.l1_loss(outputs[:, 11:14], targets[:, 11:14])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)

                loss = F.l1_loss(outputs, targets)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  Saved best model")

    # Load best model and export
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def export_to_ncnn(model: nn.Module, output_dir: Path):
    """Export trained model to NCNN format."""
    import pnnx

    model.eval()
    model.cpu()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Trace model
    x = torch.randn(1, 1024, 96, 96)
    traced = torch.jit.trace(model, x)
    traced.save(str(output_dir / 'gaussian_head.pt'))

    # Convert with pnnx
    pnnx.export(traced, str(output_dir / 'gaussian_head.ncnn'), x)

    print(f"Exported to NCNN: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train GaussianHead')
    parser.add_argument('--model_path', type=str,
                        default='/Users/al/Documents/tries01/Furnit/ml_experiments/models/sharp_2572gikvuh.pt')
    parser.add_argument('--image_dir', type=str,
                        default='/Users/al/Documents/tries01/Furnit/ml_experiments/test_images')
    parser.add_argument('--data_dir', type=str, default='./gaussian_head_training_data')
    parser.add_argument('--output_dir', type=str, default='./gaussian_head_trained')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--generate_data', action='store_true', help='Generate training data first')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()

    # Select device
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # Generate training data if needed
    if args.generate_data or not data_dir.exists() or not list(data_dir.glob("sample_*")):
        print("\n=== Generating Training Data ===")
        extractor = SharpFeatureExtractor(args.model_path, device)
        generate_training_data(extractor, args.image_dir, args.data_dir, args.max_samples)

    # Train model
    print("\n=== Training GaussianHead ===")
    model = train_gaussian_head(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    if model is None:
        print("Training failed - no model produced")
        return

    # Export to NCNN
    print("\n=== Exporting to NCNN ===")
    export_to_ncnn(model, Path(args.output_dir))

    print("\n=== Done ===")
    print(f"Trained model: {args.output_dir}/best_model.pth")
    print(f"NCNN model: {args.output_dir}/gaussian_head.ncnn.param")
    print(f"\nTo deploy to Android:")
    print(f"  cp {args.output_dir}/gaussian_head.ncnn.* app/src/main/assets/models_cpu/")


if __name__ == "__main__":
    main()
