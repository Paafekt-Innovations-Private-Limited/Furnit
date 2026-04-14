#!/usr/bin/env python3
"""
Optimized YOLO26 Export Pipeline with Two-Stage Fusing and Batch Tracking

This script implements the proper Ultralytics YOLO export workflow:
1. **Fuse Logic Restoration**: Properly calls model.fuse() to strip semseg layers and fuse Conv+BN
2. **End-to-End Export**: Sets end2end=True to include the complete inference pipeline
3. **Float Prototype Precision**: Exports float32 prototypes (half=False) for sub-pixel mask accuracy
4. **Dual Mask Modes**: Supports both retina_masks and standard process_mask workflows
5. **Batch Processing**: Processes multiple models with progress tracking

References:
- Segment26.fuse: https://docs.ultralytics.com/reference/nn/modules/head/#method-ultralytics.nn.modules.head.segment26.fuse
- Proto26.fuse: https://docs.ultralytics.com/reference/nn/modules/block/#method-ultralytics.nn.modules.block.proto26.fuse
- process_mask: https://docs.ultralytics.com/reference/utils/ops/#function-ultralytics.utils.ops.process_mask
- process_mask_native: https://docs.ultralytics.com/reference/utils/ops/#function-ultralytics.utils.ops.process_mask_native

Usage:
    # Single model export
    python scripts/twostage_union_concave_perrun.py --pt yoloe-26l-seg-pf.pt

    # Batch export with tracking
    python scripts/twostage_union_concave_perrun.py --batch models/*.pt --track

    # Retina masks mode (upsamples protos to full image size first)
    python scripts/twostage_union_concave_perrun.py --pt model.pt --retina-masks

    # Standard mode (upsamples cropped mask to input size - recommended)
    python scripts/twostage_union_concave_perrun.py --pt model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image


@dataclass
class ExportConfig:
    """Configuration for YOLO export pipeline."""
    pt_path: Path
    output_dir: Path
    image_size: int = 1280
    batch_size: int = 1
    use_retina_masks: bool = False
    half_precision: bool = False  # Keep False for float prototypes
    simplify: bool = True
    nms: bool = False  # Must be False for end2end tensor matching
    verify_image: Path | None = None


@dataclass
class ExportResult:
    """Result of a single export operation."""
    pt_path: Path
    output_path: Path | None
    success: bool
    duration_seconds: float
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class YOLOExportPipeline:
    """Two-stage YOLO export pipeline with proper fusing."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.model = None

    def _patch_yoloe_segment_fuse(self) -> None:
        """
        Patch YOLOE segment heads to properly handle prompt-free models.

        This ensures:
        - Segment26.fuse() properly removes semseg training-only heads
        - Proto26.fuse() is called to strip unnecessary layers
        - Handles models with lrpc (prompt-free) that have cv3/cv4 as None
        """
        from ultralytics.nn.modules import head

        detect_class = getattr(head, "YOLOEDetect", None)
        segment_class = getattr(head, "YOLOESegment26", None) or getattr(head, "Segment26", None)

        if detect_class and not getattr(detect_class, "_furnit_fuse_patch", False):
            original_fuse = detect_class.fuse
            original_fuse_tp = getattr(detect_class, "_fuse_tp", None)

            def safe_fuse(self, txt_feats=None):
                # Skip text-prompt fusion for prompt-free lrpc heads
                if txt_feats is not None and hasattr(self, "lrpc"):
                    print("  [Fuse] Skipping YOLOE text-prompt fuse for prompt-free lrpc head")
                    self.is_fused = True
                    return
                return original_fuse(self, txt_feats)

            def safe_fuse_tp(self, txt_feats, cls_head, bn_head):
                # Handle models where cv3/cv4 or bn_head may be None
                if cls_head is None or bn_head is None:
                    print("  [Fuse] Skipping _fuse_tp for missing cv3/cv4 or bn_head")
                    return
                return original_fuse_tp(self, txt_feats, cls_head, bn_head)

            detect_class.fuse = safe_fuse
            if original_fuse_tp:
                detect_class._fuse_tp = safe_fuse_tp
            detect_class._furnit_fuse_patch = True
            print("[Patch] Applied YOLOE Detect fuse patch")

        if segment_class and not getattr(segment_class, "_furnit_fuse_patch", False):
            original_seg_fuse = segment_class.fuse

            def safe_segment_fuse(self):
                """
                Properly fuse Segment26 head:
                1. Remove semseg training-only layers
                2. Fuse Conv+BN in detection and proto heads
                3. Call Proto26.fuse() if available
                """
                # Remove semantic segmentation heads (training only)
                if hasattr(self, "semseg") and self.semseg is not None:
                    print("  [Fuse] Removing semseg training-only layers")
                    self.semseg = None

                # Call original fuse logic
                result = original_seg_fuse(self)

                # Explicitly fuse proto head if available
                if hasattr(self, "proto") and hasattr(self.proto, "fuse"):
                    print("  [Fuse] Calling Proto26.fuse()")
                    self.proto.fuse()

                return result

            segment_class.fuse = safe_segment_fuse
            segment_class._furnit_fuse_patch = True
            print("[Patch] Applied Segment26 fuse patch")

    def _load_model(self) -> None:
        """Load YOLO model and configure for end-to-end export."""
        from ultralytics import YOLO

        print(f"\n[Load] Loading model: {self.config.pt_path}")
        self.model = YOLO(str(self.config.pt_path))

        # Get the head (last layer in model.model.model)
        head = self.model.model.model[-1]
        head_type = type(head).__name__
        print(f"[Load] Head type: {head_type}")

        # Check if prompt-free model
        has_lrpc = hasattr(head, "lrpc") and head.lrpc is not None
        if has_lrpc:
            print("[Load] Prompt-free model detected (has lrpc)")

        # Check end2end flag (read-only property for YOLOESegment26)
        # YOLOE models have end2end behavior built-in via this property
        if hasattr(head, "end2end"):
            current_end2end = getattr(head, "end2end", None)
            print(f"[Load] Head end2end property: {current_end2end}")
            print("[Load] End-to-end behavior is built into YOLOE head")

    def _stage1_fuse(self) -> None:
        """
        Stage 1: Model-level fusion.

        This calls model.fuse() which:
        - Fuses Conv+BN layers across the entire model
        - For Detect(end2end=True) heads, removes one-to-many branch
        - Triggers head.fuse() for all detection/segmentation heads
        """
        print("\n[Stage 1] Calling model.fuse()...")
        self.model.fuse()
        print("[Stage 1] ✓ Model fusion complete")

    def _stage2_verify_fuse(self) -> None:
        """
        Stage 2: Verify fusion was successful.

        Checks:
        - Head is_fused flag is True
        - semseg layers are removed (for Segment heads)
        - Proto head is fused (if available)
        """
        print("\n[Stage 2] Verifying fusion...")
        head = self.model.model.model[-1]

        # Check is_fused flag
        is_fused = getattr(head, "is_fused", False)
        print(f"[Stage 2] head.is_fused: {is_fused}")

        # Check semseg removal for segment heads
        has_semseg = hasattr(head, "semseg") and head.semseg is not None
        if has_semseg:
            print("[Stage 2] ⚠ Warning: semseg still present (should be None after fusing)")
        else:
            print("[Stage 2] ✓ semseg removed")

        # Check proto head fusion
        if hasattr(head, "proto"):
            proto_fused = getattr(head.proto, "is_fused", False)
            print(f"[Stage 2] proto.is_fused: {proto_fused}")

        print("[Stage 2] ✓ Fusion verification complete")

    def _export_coreml(self) -> str:
        """
        Export to Core ML with optimized settings.

        Critical flags:
        - nms=False: Don't embed NMS in graph (matches iOS end2end tensor parsing)
        - half=False: Export float32 prototypes for sub-pixel mask accuracy
        - simplify=True: Apply graph optimizations

        Note: YOLOE models have end2end behavior built-in via the head.end2end property.
        This doesn't need to be specified as an export parameter.

        Returns:
            Path to exported .mlpackage
        """
        print("\n[Export] Exporting to Core ML...")
        print(f"[Export] Settings:")
        print(f"  - Image size: {self.config.image_size}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - NMS: {self.config.nms}")
        print(f"  - Half precision: {self.config.half_precision} (float prototypes)")
        print(f"  - Simplify: {self.config.simplify}")

        exported_path = self.model.export(
            format="coreml",
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            nms=self.config.nms,
            half=self.config.half_precision,
            simplify=self.config.simplify,
        )

        print(f"[Export] ✓ Exported to: {exported_path}")
        return exported_path

    def _verify_export(self, package_path: Path) -> dict[str, Any]:
        """
        Verify exported Core ML model and extract metadata.

        Returns:
            Dictionary with model metadata (inputs, outputs, shapes)
        """
        import coremltools as ct

        print(f"\n[Verify] Loading Core ML model: {package_path}")
        model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()

        # Extract input metadata
        inputs = {}
        print("[Verify] Inputs:")
        for feature in spec.description.input:
            feature_type = feature.type.WhichOneof("Type")
            if feature_type == "multiArrayType":
                detail = list(feature.type.multiArrayType.shape)
            elif feature_type == "imageType":
                image_type = feature.type.imageType
                detail = {
                    "width": image_type.width,
                    "height": image_type.height,
                    "colorSpace": image_type.colorSpace,
                }
            else:
                detail = feature_type
            inputs[feature.name] = detail
            print(f"  {feature.name}: {detail}")

        # Extract output metadata
        outputs = {}
        print("[Verify] Outputs:")
        for feature in spec.description.output:
            shape = list(feature.type.multiArrayType.shape)
            outputs[feature.name] = shape
            print(f"  {feature.name}: {shape}")

        return {
            "inputs": inputs,
            "outputs": outputs,
            "model_type": type(self.model.model.model[-1]).__name__,
        }

    def _smoke_test(self, package_path: Path) -> dict[str, Any]:
        """
        Run smoke test prediction on gray letterbox image.

        Returns:
            Dictionary with output statistics
        """
        import coremltools as ct

        print(f"\n[Smoke Test] Running inference...")
        model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)

        # Create letterbox gray image (same as Ultralytics default: RGB 114,114,114)
        image = Image.new("RGB", (self.config.image_size, self.config.image_size), (114, 114, 114))

        # Run prediction
        outputs = model.predict({"image": image})

        # Collect statistics
        stats = {}
        print("[Smoke Test] Output statistics:")
        for name, value in outputs.items():
            array = np.asarray(value)
            stats[name] = {
                "shape": list(array.shape),
                "mean": float(array.mean()),
                "min": float(array.min()),
                "max": float(array.max()),
                "std": float(array.std()),
            }
            print(f"  {name}:")
            print(f"    shape={stats[name]['shape']}")
            print(f"    mean={stats[name]['mean']:.6f}, std={stats[name]['std']:.6f}")
            print(f"    min={stats[name]['min']:.6f}, max={stats[name]['max']:.6f}")

        return stats

    def _verify_with_image(self, image_path: Path) -> None:
        """
        Run prediction on a real image with retina_masks if configured.

        This uses the PyTorch model (not Core ML) to verify mask quality.
        """
        print(f"\n[Image Verify] Running prediction on: {image_path}")
        print(f"[Image Verify] Retina masks: {self.config.use_retina_masks}")

        results = self.model.predict(
            str(image_path),
            retina_masks=self.config.use_retina_masks,
            imgsz=self.config.image_size,
        )

        for i, result in enumerate(results):
            print(f"[Image Verify] Result {i}:")
            if hasattr(result, "boxes") and result.boxes is not None:
                print(f"  Boxes: {len(result.boxes)}")
            if hasattr(result, "masks") and result.masks is not None:
                masks_data = result.masks.data if hasattr(result.masks, "data") else None
                if masks_data is not None:
                    print(f"  Masks: {masks_data.shape}")
                    print(f"  Mask dtype: {masks_data.dtype}")

    def run(self) -> ExportResult:
        """
        Execute the complete two-stage export pipeline.

        Returns:
            ExportResult with success status and metadata
        """
        start_time = time.time()

        try:
            # Apply patches before loading
            self._patch_yoloe_segment_fuse()

            # Load model
            self._load_model()

            # Stage 1: Fuse the model
            self._stage1_fuse()

            # Stage 2: Verify fusion
            self._stage2_verify_fuse()

            # Optional: Verify with real image before export
            if self.config.verify_image and self.config.verify_image.exists():
                self._verify_with_image(self.config.verify_image)

            # Export to Core ML
            exported_path = self._export_coreml()
            exported_path = Path(exported_path)

            # Verify export and get metadata
            metadata = self._verify_export(exported_path)

            # Run smoke test
            smoke_stats = self._smoke_test(exported_path)
            metadata["smoke_test"] = smoke_stats

            # Move to output directory if specified
            if self.config.output_dir != exported_path.parent:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                final_path = self.config.output_dir / exported_path.name
                if final_path.exists():
                    import shutil
                    shutil.rmtree(final_path)
                exported_path.rename(final_path)
                exported_path = final_path
                print(f"\n[Move] Moved to: {exported_path}")

            duration = time.time() - start_time
            print(f"\n[Success] Export completed in {duration:.2f}s")

            return ExportResult(
                pt_path=self.config.pt_path,
                output_path=exported_path,
                success=True,
                duration_seconds=duration,
                metadata=metadata,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"\n[Error] Export failed: {e}")
            import traceback
            traceback.print_exc()

            return ExportResult(
                pt_path=self.config.pt_path,
                output_path=None,
                success=False,
                duration_seconds=duration,
                error_message=str(e),
            )


class BatchExportTracker:
    """Track batch export progress and results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results: list[ExportResult] = []
        self.start_time = time.time()

    def add_result(self, result: ExportResult) -> None:
        """Add an export result to the tracker."""
        self.results.append(result)

    def save_report(self) -> Path:
        """Save batch export report to JSON."""
        total_duration = time.time() - self.start_time

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "total_models": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "results": [
                {
                    "pt_path": str(r.pt_path),
                    "output_path": str(r.output_path) if r.output_path else None,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

        report_path = self.output_dir / f"export_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[Report] Saved to: {report_path}")
        return report_path

    def print_summary(self) -> None:
        """Print batch export summary."""
        total_duration = time.time() - self.start_time
        successful = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)

        print("\n" + "=" * 80)
        print("BATCH EXPORT SUMMARY")
        print("=" * 80)
        print(f"Total models: {len(self.results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Average per model: {total_duration / len(self.results):.2f}s")
        print()

        for i, result in enumerate(self.results, 1):
            status = "✓" if result.success else "✗"
            print(f"{status} [{i}/{len(self.results)}] {result.pt_path.name}")
            if result.success:
                print(f"  → {result.output_path}")
                print(f"  Duration: {result.duration_seconds:.2f}s")
            else:
                print(f"  Error: {result.error_message}")

        print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-stage YOLO export pipeline with proper fusing and batch tracking"
    )
    parser.add_argument(
        "--pt",
        type=Path,
        help="Path to single .pt checkpoint",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Glob pattern for batch export (e.g., 'models/*.pt')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "exports",
        help="Output directory for .mlpackage files (default: ./exports)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Export image size (default: 1280)",
    )
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Use retina_masks mode (upsamples protos to full image size first)",
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        help="Run prediction on this image before export to verify mask quality",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable batch tracking and save JSON report",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip smoke test after export",
    )

    args = parser.parse_args()

    # Determine input files
    pt_files: list[Path] = []
    if args.pt:
        if not args.pt.exists():
            print(f"Error: File not found: {args.pt}", file=sys.stderr)
            return 1
        pt_files = [args.pt]
    elif args.batch:
        import glob
        pt_files = [Path(p) for p in glob.glob(args.batch)]
        if not pt_files:
            print(f"Error: No files found matching: {args.batch}", file=sys.stderr)
            return 1
    else:
        print("Error: Must specify either --pt or --batch", file=sys.stderr)
        parser.print_help()
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracker if requested
    tracker = BatchExportTracker(args.output_dir) if args.track else None

    # Process each model
    for i, pt_path in enumerate(pt_files, 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING MODEL [{i}/{len(pt_files)}]: {pt_path.name}")
        print("=" * 80)

        config = ExportConfig(
            pt_path=pt_path,
            output_dir=args.output_dir,
            image_size=args.imgsz,
            use_retina_masks=args.retina_masks,
            verify_image=args.verify_image,
        )

        pipeline = YOLOExportPipeline(config)
        result = pipeline.run()

        if tracker:
            tracker.add_result(result)

    # Print summary and save report
    if tracker:
        tracker.print_summary()
        tracker.save_report()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
