"""Validate the native 2DGS backends shipped by this release."""
import os


def validate_native_backend(modelset):
    backend = str(getattr(modelset, "rasterizer", "2dgs"))
    if backend not in {"2dgs", "2dgs_3ch"}:
        raise ValueError(
            f"Unsupported rasterizer {backend!r}: this LoFT-GS release ships native "
            "2DGS only (2dgs or 2dgs_3ch). Legacy 3DGS/gsplat checkpoints require "
            "their original source revision."
        )
    if getattr(modelset, "use_hgs", False) or getattr(modelset, "use_hgs_finetune", False):
        raise ValueError("HGS is not supported by the native LoFT-GS release.")
    if os.environ.get("GS3_2DGS_USE_LIFTED_SHADOW", "0") == "1":
        raise ValueError(
            "GS3_2DGS_USE_LIFTED_SHADOW selects a retired 3DGS backend; "
            "unset it to use the native 2DGS shadow pass."
        )
    return backend
