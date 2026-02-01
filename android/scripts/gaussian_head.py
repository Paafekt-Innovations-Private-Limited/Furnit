"""
GaussianHead - Lightweight depth/position prediction head for SHARP

This module is trained offline using the full SHARP decoder as a teacher.
Once trained, it replaces the heavy decoder on mobile devices.

Architecture:
- Input: Merged encoder features [B, 1024, 96, 96] from NCNN encoder
- Output: Per-pixel 3D positions [B, 3, H, W] + opacity [B, 1, H, W]

Training workflow:
1. Run full SHARP (PyTorch) on dataset of images
2. Save encoder features + decoder outputs as training pairs
3. Train this lightweight head to mimic decoder
4. Export to ONNX -> NCNN for mobile deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """
    Lightweight head to predict 3D Gaussian positions from encoder features.

    Much smaller than full SHARP decoder (~2MB vs ~35MB+).
    Designed for mobile deployment via NCNN.
    """

    def __init__(self,
                 in_channels: int = 1024,
                 hidden_channels: int = 256,
                 output_size: int = 384,
                 predict_opacity: bool = True,
                 predict_scale: bool = False):
        super().__init__()

        self.output_size = output_size
        self.predict_opacity = predict_opacity
        self.predict_scale = predict_scale

        # Channel reduction (1024 -> 256)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        # Upsample path: 96 -> 192 -> 384
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
        )

        # Output channels: xyz (3) + optional opacity (1) + optional scale (3)
        out_channels = 3
        if predict_opacity:
            out_channels += 1
        if predict_scale:
            out_channels += 3

        # Position/attribute prediction head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: Merged encoder features [B, 1024, 96, 96]

        Returns:
            dict with:
                - positions: [B, 3, H, W] xyz coordinates
                - opacity: [B, 1, H, W] if predict_opacity
                - scale: [B, 3, H, W] if predict_scale
        """
        # Reduce channels
        x = self.reduce(features)  # [B, 256, 96, 96]

        # Upsample
        x = self.up1(x)  # [B, 256, 192, 192]
        x = self.up2(x)  # [B, 128, 384, 384]

        # Predict attributes
        out = self.head(x)  # [B, out_channels, 384, 384]

        # Parse outputs
        result = {}
        idx = 0

        # Positions (xyz) - use tanh to bound to [-1, 1] then scale
        result['positions'] = torch.tanh(out[:, idx:idx+3]) * 2.0  # [-2, 2] range
        idx += 3

        if self.predict_opacity:
            result['opacity'] = torch.sigmoid(out[:, idx:idx+1])
            idx += 1

        if self.predict_scale:
            # Scales should be positive, use softplus
            result['scale'] = F.softplus(out[:, idx:idx+3]) * 0.01 + 0.001
            idx += 3

        return result


class GaussianHeadLoss(nn.Module):
    """
    Loss function for training GaussianHead.

    Combines:
    - Position MSE loss
    - Optional opacity BCE loss
    - Optional scale loss
    """

    def __init__(self,
                 position_weight: float = 1.0,
                 opacity_weight: float = 0.1,
                 scale_weight: float = 0.1):
        super().__init__()
        self.position_weight = position_weight
        self.opacity_weight = opacity_weight
        self.scale_weight = scale_weight

    def forward(self, pred: dict, target: dict) -> dict:
        losses = {}

        # Position loss (L1 + L2)
        pos_l1 = F.l1_loss(pred['positions'], target['positions'])
        pos_l2 = F.mse_loss(pred['positions'], target['positions'])
        losses['position'] = pos_l1 + 0.5 * pos_l2

        # Opacity loss
        if 'opacity' in pred and 'opacity' in target:
            losses['opacity'] = F.binary_cross_entropy(
                pred['opacity'], target['opacity']
            )

        # Scale loss
        if 'scale' in pred and 'scale' in target:
            losses['scale'] = F.l1_loss(pred['scale'], target['scale'])

        # Total weighted loss
        total = self.position_weight * losses['position']
        if 'opacity' in losses:
            total += self.opacity_weight * losses['opacity']
        if 'scale' in losses:
            total += self.scale_weight * losses['scale']

        losses['total'] = total
        return losses


def export_to_onnx(model: GaussianHead,
                   output_path: str,
                   input_size: tuple = (1, 1024, 96, 96)):
    """Export trained model to ONNX for NCNN conversion."""
    model.eval()

    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['encoder_features'],
        output_names=['positions', 'opacity'] if model.predict_opacity else ['positions'],
        dynamic_axes={
            'encoder_features': {0: 'batch'},
            'positions': {0: 'batch'},
        },
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"Exported to {output_path}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = GaussianHead(
        in_channels=1024,
        hidden_channels=256,
        output_size=384,
        predict_opacity=True,
        predict_scale=False,
    )

    # Print model size
    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params * 4 / 1024 / 1024:.2f} MB)")

    # Test forward pass
    x = torch.randn(1, 1024, 96, 96)
    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output positions shape: {out['positions'].shape}")
    if 'opacity' in out:
        print(f"Output opacity shape: {out['opacity'].shape}")

    # Export to ONNX
    export_to_onnx(model, "gaussian_head.onnx")
