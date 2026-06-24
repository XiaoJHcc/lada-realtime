# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# Lightweight engine-path helpers for the BasicVSR++ TensorRT sub-engines.
# No torch / tensorrt imports. Ported from jasna's engine_paths.py, keeping
# only the BasicVSR++ helpers (the unet/sd15/yolo/_frozen parts are dropped).
#
# Engines live in ``<weights dir>/<weights stem>_sub_engines/`` next to the
# .pth weights — i.e. inside model_weights/ — so they are cached per weight
# file, precision, system, and clip-size upper bound (all encoded in the
# filename). Changing any of those triggers a recompile, which is intended:
# a TRT engine is bound to its GPU architecture, precision, and shape bounds.
from __future__ import annotations

import os

BASICVSRPP_DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")


def engine_system_suffix() -> str:
    return ".win" if os.name == "nt" else ".linux"


def engine_precision_name(*, fp16: bool) -> str:
    return "fp16" if bool(fp16) else "fp32"


def _basicvsrpp_sub_engine_dir(model_weights_path: str) -> str:
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def get_basicvsrpp_sub_engine_paths(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> dict[str, str]:
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        paths[f"loop_body_{d}"] = os.path.join(engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine")
    paths["preprocess"] = os.path.join(engine_dir, f"preprocess_b{max_clip_size}.trt_{prec}{suf}.engine")
    paths["upsample"] = os.path.join(engine_dir, f"upsample_dyn_b{max_clip_size}.trt_{prec}{suf}.engine")
    return paths


def all_basicvsrpp_sub_engines_exist(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> bool:
    return all(os.path.isfile(p) for p in get_basicvsrpp_sub_engine_paths(model_weights_path, fp16, max_clip_size).values())
