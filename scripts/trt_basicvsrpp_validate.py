# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Standalone validation of the BasicVSR++ TensorRT split port.

Loads the real generic_v1.2 weights, compiles the 6 sub-engines, builds the
split forward, and compares its output against the PyTorch reference on a clip
shaped like what MosaicDetector emits (256x256, BGR-ish content range [0,1]).

This is the "this step passing == port succeeded" gate from the design doc.
Run: .venv/Scripts/python.exe scripts/trt_basicvsrpp_validate.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

from lada import ModelFiles
from lada.models.basicvsrpp.inference import load_model
from lada.restorationpipeline.basicvsrpp_sub_engines import (
    compile_basicvsrpp_sub_engines,
    create_split_forward,
)

DEVICE = torch.device("cuda")
FP16 = True
MAX_CLIP_SIZE = int(os.environ.get("VALIDATE_MAX_CLIP", "30"))  # engine upper bound to compile
TEST_T = int(os.environ.get("VALIDATE_T", "16"))                # frames in the test clip
SIZE = 256


def main() -> int:
    path = ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2").path
    print(f"weights: {path}")

    print("loading model (fp16)...")
    model = load_model(None, path, DEVICE, FP16)

    print(f"compiling 6 sub-engines (max_clip_size={MAX_CLIP_SIZE})...")
    t0 = time.time()
    compile_basicvsrpp_sub_engines(
        model=model, device=DEVICE, fp16=FP16,
        model_weights_path=path, max_clip_size=MAX_CLIP_SIZE,
        optimization_level=5,
    )
    print(f"compile took {time.time() - t0:.1f}s")

    split = create_split_forward(model, path, DEVICE, FP16, max_clip_size=MAX_CLIP_SIZE)
    assert split is not None, "create_split_forward returned None (engines missing?)"

    # A clip shaped like the restorer feeds: (N=1, T, C=3, H, W) float16 in [0,1]
    torch.manual_seed(0)
    lqs = torch.rand(1, TEST_T, 3, SIZE, SIZE, dtype=torch.float16, device=DEVICE)

    with torch.inference_mode():
        ref = model(inputs=lqs)
        # warm + time TRT
        trt_out = split(lqs)
        torch.cuda.synchronize()

        # numeric parity
        ref_f = ref.float()
        trt_f = trt_out.float()
        assert ref_f.shape == trt_f.shape, f"shape mismatch {ref_f.shape} vs {trt_f.shape}"
        abs_err = (ref_f - trt_f).abs()
        mae = abs_err.mean().item()
        maxe = abs_err.max().item()
        # PSNR over [0,1]
        mse = (abs_err ** 2).mean().item()
        psnr = float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    print(f"shape: {tuple(trt_f.shape)}")
    print(f"MAE={mae:.6f}  max_abs={maxe:.6f}  PSNR={psnr:.2f} dB")

    # throughput
    def bench(fn, n=20):
        with torch.inference_mode():
            torch.cuda.synchronize(); s = time.time()
            for _ in range(n):
                fn(lqs)
            torch.cuda.synchronize()
        return (time.time() - s) / n

    t_ref = bench(lambda x: model(inputs=x))
    t_trt = bench(lambda x: split(x))
    print(f"forward/clip(T={TEST_T}): PyTorch={t_ref*1000:.1f}ms  TRT={t_trt*1000:.1f}ms  speedup={t_ref/t_trt:.2f}x")
    print(f"  fps: PyTorch={TEST_T/t_ref:.1f}  TRT={TEST_T/t_trt:.1f}")

    # gate: design doc tolerance MAE < 2/255 ≈ 0.0078
    ok = mae < (2.0 / 255.0)
    print("PARITY GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
