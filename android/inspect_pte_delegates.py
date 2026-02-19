#!/usr/bin/env python3
"""
Inspect a .pte file to see if XNNPACK/Vulkan delegates are used.

Usage:
  python inspect_pte_delegates.py executorch_models/sharp_split_part1.pte

This helps answer: "Is XNNPACK really being used?"
"""

import sys
from pathlib import Path


def inspect_pte(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File: {path.name} ({size_mb:.1f} MB)")
    print("=" * 60)

    # Method 1: Grep binary for delegate identifiers
    data = path.read_bytes()
    xnnpack_hits = data.count(b"xnnpack") + data.count(b"XNNPACK")
    vulkan_hits = data.count(b"vulkan") + data.count(b"Vulkan")
    portable_hits = data.count(b"portable") + data.count(b"Portable")
    # Backend IDs in ExecuTorch flatbuffer
    backend_ids = [
        (b"xnnpack_backend", "XNNPACK backend"),
        (b"XnnpackBackend", "XNNPACK backend (camelCase)"),
        (b"vulkan_backend", "Vulkan backend"),
        (b"VulkanBackend", "Vulkan backend"),
    ]
    print("\nBinary string search:")
    for needle, label in backend_ids:
        if needle in data:
            print(f"  FOUND: {label} ({needle!r})")
    if xnnpack_hits > 0:
        print(f"  'xnnpack'/'XNNPACK' occurrences: {xnnpack_hits}")
    if vulkan_hits > 0:
        print(f"  'vulkan'/'Vulkan' occurrences: {vulkan_hits}")
    if xnnpack_hits == 0 and vulkan_hits == 0:
        print("  No obvious XNNPACK/Vulkan strings in binary (may be normal)")

    # Method 2: Load with ExecuTorch runtime and try to dump program
    try:
        import executorch
        from executorch.runtime import Runtime

        print("\nExecuTorch runtime inspection:")
        runtime = Runtime.get()
        program = runtime.load_program(path)
        method_names = program.method_names
        print(f"  Methods: {method_names}")

        # Try to get program/execution plan info
        if hasattr(program, "execution_plan"):
            ep = program.execution_plan
            if ep is not None and hasattr(ep, "operators"):
                ops = list(ep.operators) if hasattr(ep.operators, "__iter__") else []
                delegate_calls = [o for o in ops if "delegate" in str(o).lower()]
                if delegate_calls:
                    print(f"  Delegate calls in plan: {len(delegate_calls)}")
                    for d in delegate_calls[:5]:
                        print(f"    - {d}")
                else:
                    print("  (Could not enumerate operators)")

        # Check ExecuTorch version
        et_version = getattr(executorch, "__version__", "unknown")
        print(f"  ExecuTorch version: {et_version}")

    except ImportError as e:
        print(f"\nExecuTorch not available: {e}")
        print("  pip install executorch")
    except Exception as e:
        print(f"\nRuntime load/dump failed: {e}")

    # Method 3: Export script reminder
    print("\n" + "=" * 60)
    print("Reminder: Backend is chosen at EXPORT time:")
    print("  python export_sharp_executorch_split4.py --backend xnnpack   # XNNPACK")
    print("  python export_sharp_executorch_split4.py --backend vulkan    # Vulkan GPU")
    print("  python export_sharp_executorch_split4.py --backend portable  # CPU fallback")
    print("\nIf you exported with --backend xnnpack, the .pte contains XNNPACK")
    print("partitioned subgraphs. ViT attention ops may still run on portable")
    print("(CPU), so ~2s/patch can mean partial XNNPACK usage.")


def main():
    if len(sys.argv) < 2:
        pte = Path(__file__).parent / "executorch_models" / "sharp_split_part1.pte"
        if pte.exists():
            print(f"Using default: {pte}")
        else:
            print("Usage: python inspect_pte_delegates.py <path/to/model.pte>")
            sys.exit(1)
    else:
        pte = Path(sys.argv[1])

    inspect_pte(pte)


if __name__ == "__main__":
    main()
