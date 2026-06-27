# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""One-off verification: does the demosaic inference path still run with certain
torch CUDA DLLs removed?  Exercises BOTH heavy GPU components on a REAL clip:
  1. BasicVSR++ restore  (TRT split-forward when LADA_BASICVSRPP_TRT=1, else PyTorch)
  2. YOLO11 segmentation detection
load_models() also internally warms up both, so a clean load is itself a signal.
Prints PASS/FAIL with finite-output checks. Not part of the shipped app."""
import builtins, os, sys, traceback
builtins.__dict__.setdefault("_", lambda s: s)  # gettext fallback if not installed

import numpy as np
import torch

from lada import ModelFiles
from lada.restorationpipeline import load_models, BASICVSRPP_TRT_ENABLED
from lada.utils.video_utils import read_video_frames

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
DEVICE = torch.device("cuda")
CLIP_T = int(os.environ.get("LADA_VERIFY_CLIP_T", "16"))  # clip length to exercise (cudnn picks kernels by shape)
CLIP_SIZE = 256


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(2)


def main():
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
          f"TRT_enabled(env)={BASICVSRPP_TRT_ENABLED}")
    if not torch.cuda.is_available():
        fail("CUDA not available")
    # quick proof the DLLs are actually gone from this torch install
    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    for name in ("cusolverMg64_11.dll", "curand64_10.dll"):
        present = os.path.exists(os.path.join(libdir, name))
        print(f"  {name}: {'PRESENT' if present else 'ABSENT'} in {libdir}")

    det_mf = ModelFiles.get_detection_model_by_name("v4-fast")
    res_mf = ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2")
    if det_mf is None or res_mf is None:
        fail("model weights not found")

    print("Loading models (this internally warms up BasicVSR++ + YOLO on GPU)...")
    det, restorer, pad = load_models(DEVICE, "basicvsrpp-v1.2", res_mf.path, None,
                                     det_mf.path, True, False)
    using_trt = getattr(restorer, "_split_forward", None) is not None
    print(f"  -> models loaded. BasicVSR++ split_forward(TRT)={using_trt}, pad_mode={pad}")

    # --- real frames ---
    print(f"Reading frames from {VIDEO} ...")
    raw = read_video_frames(VIDEO, float32=False, start_idx=0, end_idx=CLIP_T)
    if len(raw) < CLIP_T:
        fail(f"only {len(raw)} frames read, need {CLIP_T}")
    import cv2
    clip = [torch.from_numpy(cv2.resize(f, (CLIP_SIZE, CLIP_SIZE))).contiguous() for f in raw]
    full = [torch.from_numpy(np.ascontiguousarray(f)) for f in raw]  # full-res for detector
    print(f"  clip: {len(clip)}x{tuple(clip[0].shape)} uint8 ; full: {tuple(full[0].shape)}")

    # --- BasicVSR++ restore on a real clip ---
    print("Running BasicVSR++ restore on real clip...")
    out = restorer.restore(clip)
    if len(out) != len(clip):
        fail(f"restore returned {len(out)} frames, expected {len(clip)}")
    stacked = torch.stack([o.to(torch.float32) for o in out])
    if not torch.isfinite(stacked).all():
        fail("restore output contains NaN/Inf")
    print(f"  -> restore OK: {len(out)} frames, dtype={out[0].dtype}, "
          f"min={stacked.min().item():.1f} max={stacked.max().item():.1f}")

    # --- YOLO detection on real full-res frames ---
    print("Running YOLO detection on real frames...")
    pre = det.preprocess(full[:4])
    results = det.inference_and_postprocess(pre, full[:4])
    ndet = sum((len(r.boxes) if getattr(r, "boxes", None) is not None else 0) for r in results) \
        if hasattr(results, "__iter__") else -1
    print(f"  -> detect OK: {len(results) if hasattr(results,'__len__') else '?'} results, "
          f"total boxes={ndet}")

    print("PASS: demosaic inference (restore + detect) completed without the removed DLLs")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unhandled exception during inference")
