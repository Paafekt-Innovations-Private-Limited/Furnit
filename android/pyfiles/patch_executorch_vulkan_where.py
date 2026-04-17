#!/usr/bin/env python3
"""
Patch the installed ExecuTorch Vulkan op_registry to register aten.where.self
so the partitioner delegates attention masking to Vulkan (C++ impl already exists).

Run once from android/:  python3 patch_executorch_vulkan_where.py

Then re-export:  ./export_sharp_executorch_vulkan_full.sh
"""

import sys


def main():
    try:
        import executorch.backends.vulkan.op_registry as reg_mod
    except ImportError as e:
        print("Error: executorch not installed.", e, file=sys.stderr)
        return 1

    reg_path = reg_mod.__file__
    print(f"Patching: {reg_path}")

    with open(reg_path, "r") as f:
        content = f.read()

    # Insert before pow.Tensor_Scalar block (works across executorch versions)
    marker = "@update_features(\n    [\n        exir_ops.edge.aten.pow.Tensor_Scalar,"
    # Use minimal OpFeatures (older executorch has no inputs_dtypes/supports_highdim)
    insertion = """@update_features(exir_ops.edge.aten.where.self)
def register_where_self():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


""" + marker

    if "register_where_self" in content:
        print("Already patched (register_where_self present).")
        return 0

    if marker not in content:
        print("Could not find insertion point (BinaryScalarOp.cpp marker).", file=sys.stderr)
        return 1

    new_content = content.replace(marker, insertion)
    with open(reg_path, "w") as f:
        f.write(new_content)

    print("Patched. Verify: python3 -c \"import executorch.backends.vulkan.op_registry as r; from executorch.exir.dialects._ops import ops as o; print('where.self registered:', o.edge.aten.where.self in r.vulkan_supported_ops)\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
