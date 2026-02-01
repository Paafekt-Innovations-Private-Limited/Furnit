"""
Training script for GaussianHead.

Workflow:
1. First run generate_training_data.py to create dataset from full SHARP
2. Run this script to train the lightweight head
3. Export to ONNX, then convert to NCNN

Usage:
    python train_gaussian_head.py --data_dir ./training_data --epochs 100
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gaussian_head import GaussianHead, GaussianHeadLoss, export_to_onnx, count_parameters


class GaussianHeadDataset(Dataset):
    """
    Dataset of (encoder_features, decoder_outputs) pairs.

    Expected directory structure:
        data_dir/
            sample_0000/
                encoder_features.npy   # [1024, 96, 96]
                positions.npy          # [3, H, W] ground truth xyz
                opacity.npy            # [1, H, W] ground truth opacity (optional)
            sample_0001/
                ...
    """

    def __init__(self, data_dir: str, split: str = 'train', train_ratio: float = 0.9):
        self.data_dir = Path(data_dir)

        # Find all samples
        self.samples = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / 'encoder_features.npy').exists()
        ])

        # Split into train/val
        n_train = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]

        print(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load encoder features
        features = np.load(sample_dir / 'encoder_features.npy')
        features = torch.from_numpy(features).float()

        # Load ground truth positions
        positions = np.load(sample_dir / 'positions.npy')
        positions = torch.from_numpy(positions).float()

        target = {'positions': positions}

        # Load opacity if available
        opacity_path = sample_dir / 'opacity.npy'
        if opacity_path.exists():
            opacity = np.load(opacity_path)
            target['opacity'] = torch.from_numpy(opacity).float()

        return features, target


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(features)
        losses = criterion(outputs, targets)

        losses['total'].backward()
        optimizer.step()

        total_loss += losses['total'].item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(features)
        losses = criterion(outputs, targets)

        total_loss += losses['total'].item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train GaussianHead')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--predict_opacity', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = GaussianHeadDataset(args.data_dir, split='train')
    val_dataset = GaussianHeadDataset(args.data_dir, split='val')

    if len(train_dataset) == 0:
        print("No training data found! Run generate_training_data.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = GaussianHead(
        in_channels=1024,
        hidden_channels=args.hidden_channels,
        output_size=384,
        predict_opacity=args.predict_opacity,
    ).to(device)

    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params * 4 / 1024 / 1024:.2f} MB)")

    # Loss and optimizer
    criterion = GaussianHeadLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device) if len(val_dataset) > 0 else train_loss

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Load best model and export
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export to ONNX
    model.eval()
    model.cpu()
    onnx_path = output_dir / 'gaussian_head.onnx'
    export_to_onnx(model, str(onnx_path))

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"ONNX model saved to: {onnx_path}")
    print(f"\nTo convert to NCNN:")
    print(f"  ./ncnn/tools/onnx/onnx2ncnn {onnx_path} gaussian_head.ncnn.param gaussian_head.ncnn.bin")


if __name__ == "__main__":
    main()
