#!/usr/bin/env python3
"""
Inspect a .pte file to see if XNNPACK/Vulkan delegates are used.

Usage:
  python inspect_pte_delegates.py executorch_models/sharp_split_part1.pte
  python inspect_pte_delegates.py sharp_vulkan_only/sharp_split_part1_vulkan_fp16.pte

Verifies that Part1/Part2 were exported with the Vulkan delegate (for BackendFailed/error 32 debugging).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _top_counts(counter: Counter, limit: int = 12) -> dict[str, int]:
    return {name: count for name, count in counter.most_common(limit)}


def collect_pte_diagnostics(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    size_mb = path.stat().st_size / (1024 * 1024)
    data = path.read_bytes()
    xnnpack_hits = data.count(b"xnnpack") + data.count(b"XNNPACK")
    vulkan_hits = data.count(b"vulkan") + data.count(b"Vulkan")
    portable_hits = data.count(b"portable") + data.count(b"Portable")
    width_packed_hits = data.count(b"WIDTH_PACKED")
    channels_packed_hits = data.count(b"CHANNELS_PACKED")
    backend_ids = [
        (b"xnnpack_backend", "XNNPACK backend"),
        (b"XnnpackBackend", "XNNPACK backend (camelCase)"),
        (b"vulkan_backend", "Vulkan backend"),
        (b"VulkanBackend", "Vulkan backend"),
    ]
    vulkan_found = any(needle in data for needle in (b"vulkan_backend", b"VulkanBackend", b"Vulkan"))
    xnnpack_found = any(needle in data for needle in (b"xnnpack_backend", b"XnnpackBackend"))
    found_labels = [label for needle, label in backend_ids if needle in data]

    diagnostics = {
        "artifact": path.name,
        "path": str(path),
        "file_size_mb": round(size_mb, 3),
        "binary_search": {
            "found_labels": found_labels,
            "xnnpack_hits": xnnpack_hits,
            "vulkan_hits": vulkan_hits,
            "portable_hits": portable_hits,
            "vulkan_delegate_present": vulkan_found,
            "xnnpack_delegate_present": xnnpack_found,
        },
        "layout_string_search": {
            "width_packed_hits": width_packed_hits,
            "channels_packed_hits": channels_packed_hits,
            "high_layout_churn_suspected": bool(width_packed_hits or channels_packed_hits),
        },
    }

    try:
        from executorch.exir import schema
        from executorch.exir._serialize import _program

        pte_file = _program.deserialize_pte_binary(data)
        program = pte_file.program

        total_instructions = 0
        total_delegate_calls = 0
        total_kernel_calls = 0
        delegate_ids: Counter[str] = Counter()
        kernel_ops: Counter[str] = Counter()
        other_instruction_types: Counter[str] = Counter()
        plan_summaries = []

        for plan_idx, plan in enumerate(program.execution_plan):
            plan_instructions = 0
            plan_delegate_calls = 0
            plan_kernel_calls = 0
            plan_delegate_ids: Counter[str] = Counter()
            plan_kernel_ops: Counter[str] = Counter()
            plan_other_types: Counter[str] = Counter()

            for chain in plan.chains:
                for instruction in chain.instructions:
                    plan_instructions += 1
                    args = instruction.instr_args
                    if isinstance(args, schema.DelegateCall):
                        plan_delegate_calls += 1
                        delegate_id = f"delegate_{args.delegate_index}"
                        if 0 <= args.delegate_index < len(plan.delegates):
                            delegate_id = plan.delegates[args.delegate_index].id
                        plan_delegate_ids[delegate_id] += 1
                    elif isinstance(args, schema.KernelCall):
                        plan_kernel_calls += 1
                        op_name = f"op_{args.op_index}"
                        if 0 <= args.op_index < len(plan.operators):
                            operator = plan.operators[args.op_index]
                            op_name = operator.name if not operator.overload else f"{operator.name}.{operator.overload}"
                        plan_kernel_ops[op_name] += 1
                    else:
                        plan_other_types[type(args).__name__] += 1

            total_instructions += plan_instructions
            total_delegate_calls += plan_delegate_calls
            total_kernel_calls += plan_kernel_calls
            delegate_ids.update(plan_delegate_ids)
            kernel_ops.update(plan_kernel_ops)
            other_instruction_types.update(plan_other_types)

            plan_summaries.append(
                {
                    "plan_index": plan_idx,
                    "name": plan.name,
                    "instruction_count": plan_instructions,
                    "delegate_call_count": plan_delegate_calls,
                    "kernel_call_count": plan_kernel_calls,
                    "delegate_ids": dict(plan_delegate_ids),
                    "kernel_ops": _top_counts(plan_kernel_ops),
                    "other_instruction_types": dict(plan_other_types),
                }
            )

        diagnostics["runtime_partitioning"] = {
            "execution_plan_count": len(program.execution_plan),
            "instruction_count": total_instructions,
            "delegate_call_count": total_delegate_calls,
            "kernel_call_count": total_kernel_calls,
            "mixed_delegate_and_kernel_calls": total_delegate_calls > 0 and total_kernel_calls > 0,
            "delegate_ids": dict(delegate_ids),
            "kernel_ops": _top_counts(kernel_ops),
            "other_instruction_types": dict(other_instruction_types),
            "plan_summaries": plan_summaries,
        }
    except Exception as e:
        diagnostics["runtime_partitioning_error"] = str(e)

    return diagnostics


def inspect_pte(path: Path) -> None:
    diagnostics = collect_pte_diagnostics(path)
    if "error" in diagnostics:
        print(f"ERROR: {diagnostics['error']}")
        return

    print(f"File: {diagnostics['artifact']} ({diagnostics['file_size_mb']:.1f} MB)")
    print("=" * 60)

    print("\nBinary string search:")
    for label in diagnostics["binary_search"]["found_labels"]:
        print(f"  FOUND: {label}")
    if diagnostics["binary_search"]["xnnpack_hits"] > 0:
        print(f"  'xnnpack'/'XNNPACK' occurrences: {diagnostics['binary_search']['xnnpack_hits']}")
    if diagnostics["binary_search"]["vulkan_hits"] > 0:
        print(f"  'vulkan'/'Vulkan' occurrences: {diagnostics['binary_search']['vulkan_hits']}")
    if diagnostics["binary_search"]["xnnpack_hits"] == 0 and diagnostics["binary_search"]["vulkan_hits"] == 0:
        print("  No obvious XNNPACK/Vulkan strings in binary (may be portable-only)")

    print("\n" + "=" * 60)
    print("DELEGATE VERIFICATION:")
    if diagnostics["binary_search"]["vulkan_delegate_present"]:
        print("  VERIFIED: Vulkan delegate is present in this .pte (exported with --backend vulkan).")
        print("  If runtime still fails with forward_error=32 (BackendFailed), the issue is device/driver.")
    elif diagnostics["binary_search"]["xnnpack_delegate_present"]:
        print("  VERIFIED: XNNPACK delegate present. (Not Vulkan.)")
    else:
        print("  NOT VERIFIED as Vulkan: no Vulkan backend strings found (likely portable/CPU-only).")
        print("  For Part1 Vulkan FP16, re-export with: --backend vulkan --dtype fp16")

    print("\n" + "=" * 60)
    print("PARTITION DIAGNOSTICS:")
    runtime_partitioning = diagnostics.get("runtime_partitioning")
    if runtime_partitioning:
        print(
            "  Instructions: {instruction_count} | DelegateCall: {delegate_call_count} | KernelCall: {kernel_call_count}".format(
                **runtime_partitioning
            )
        )
        print(f"  Mixed delegate/kernel graph: {runtime_partitioning['mixed_delegate_and_kernel_calls']}")
        if runtime_partitioning["delegate_ids"]:
            print(f"  Delegates: {runtime_partitioning['delegate_ids']}")
        if runtime_partitioning["kernel_ops"]:
            print(f"  Kernel ops: {runtime_partitioning['kernel_ops']}")
    else:
        print(f"  Unavailable: {diagnostics.get('runtime_partitioning_error', 'unknown error')}")

    print("\n" + "=" * 60)
    print("Reminder: Backend is chosen at EXPORT time:")
    print("  python export_sharp_executorch_split4.py --backend vulkan --dtype fp16   # Part1 Vulkan FP16")
    print("  python export_sharp_executorch_split4.py --backend portable              # CPU fallback")
    print("  (XNNPACK removed for Android; use vulkan or portable.)")


def main():
    parser = argparse.ArgumentParser(description="Inspect ExecuTorch .pte delegate and partition diagnostics.")
    parser.add_argument("pte_path", nargs="?", help="Path to .pte model")
    parser.add_argument("--json-out", help="Optional path to write JSON diagnostics manifest")
    args = parser.parse_args()

    if not args.pte_path:
        pte = Path(__file__).parent / "executorch_models" / "sharp_split_part1.pte"
        if pte.exists():
            print(f"Using default: {pte}")
        else:
            print("Usage: python inspect_pte_delegates.py <path/to/model.pte>")
            sys.exit(1)
    else:
        pte = Path(args.pte_path)

    inspect_pte(pte)
    if args.json_out:
        diagnostics = collect_pte_diagnostics(pte)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, sort_keys=True)
        print(f"JSON diagnostics written to: {args.json_out}")


if __name__ == "__main__":
    main()
