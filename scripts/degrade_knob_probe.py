# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""
Feasibility probe for the two unverified degrade knobs:
  1. BasicVSR++ clip_size (spatial inference resolution). CLAUDE.md #2 claims compute
     scales ~quadratically with resolution (256->192 ~= -44%). UNVERIFIED. Also unknown:
     does the model even ACCEPT non-256 input, and must size be a multiple of N?
  2. YOLO detector imgsz. Detector runs ~130fps (4x headroom over 30fps need); the point
     of lowering imgsz / skipping frames is to free 4080 time for the restorer, NOT to
     make YOLO faster. Here we just measure YOLO's own throughput vs imgsz.

This measures THROUGHPUT and CAN-IT-RUN only. Quality must be judged visually on real
mosaic clips; synthetic input can't show that. Output feeds a doc update that closes
dead-end directions.

Run:
  .\.venv\Scripts\python.exe scripts\degrade_knob_probe.py
"""
import logging
import os
import time

os.environ.setdefault("LOG_LEVEL", "ERROR")

import torch

from lada.restorationpipeline import load_models
from lada import ModelFiles

logging.basicConfig(level=logging.ERROR)

DEVICE = "cuda"
FP16 = True
RESTORATION_MODEL = "basicvsrpp-generic-v1.2"
DETECTION_MODEL = "v4-fast"
CLIP_T = 30           # realtime clip length
WARMUP_ITERS = 2
MEASURE_ITERS = 6


def resolve():
    import glob
    try:
        rp = ModelFiles.get_restoration_model_by_name(RESTORATION_MODEL).path
    except Exception:
        c = glob.glob("model_weights/*restoration*v1.2*.pth"); rp = c[0] if c else None
    try:
        dp = ModelFiles.get_detection_model_by_name(DETECTION_MODEL).path
    except Exception:
        c = glob.glob("model_weights/*detection*v4*fast*.pt"); dp = c[0] if c else None
    return rp, dp


def bench_vsr(rest_model, device):
    print("\n===== BasicVSR++ clip_size sweep (forward throughput) =====")
    print(f"clip_T={CLIP_T} frames, fp16={FP16}. compute should scale ~quadratically w/ size.\n")
    base_fps = None
    for size in (256, 224, 192, 160, 128):
        if size % 4 != 0:
            print(f"  size={size}: skipped (not /4)"); continue
        try:
            clip = [torch.randint(0, 256, (size, size, 3), dtype=torch.uint8) for _ in range(CLIP_T)]
            for _ in range(WARMUP_ITERS):
                rest_model.restore([f.clone() for f in clip])
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(MEASURE_ITERS):
                rest_model.restore([f.clone() for f in clip])
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t) / MEASURE_ITERS
            fps = CLIP_T / dt
            if base_fps is None:
                base_fps = fps
            speedup = fps / base_fps
            px_ratio = (size * size) / (256 * 256)
            print(f"  size={size:>3}: {dt*1000:6.0f} ms/clip  {fps:5.1f} fps  "
                  f"x{speedup:.2f} vs 256  (pixels x{px_ratio:.2f})")
        except Exception as e:
            print(f"  size={size:>3}: FAILED -> {type(e).__name__}: {e}")


def bench_yolo(det_model, device):
    print("\n===== YOLO detector imgsz sweep (own throughput) =====")
    print("detector batch=4 like realtime. lowering imgsz frees 4080 time for restorer.\n")
    orig_imgsz = det_model.imgsz
    from ultralytics.utils.checks import check_imgsz
    for imgsz in (640, 512, 384, 320):
        try:
            det_model.imgsz = check_imgsz(imgsz, stride=det_model.stride, min_dim=2)
            # rebuild letterbox for new size
            from ultralytics.data.augment import LetterBox
            det_model.letterbox = LetterBox(det_model.imgsz, auto=True, stride=det_model.stride)
            batch = [torch.randint(0, 256, (1080, 1920, 3), dtype=torch.uint8) for _ in range(4)]
            pre = det_model.preprocess(batch)
            for _ in range(WARMUP_ITERS):
                det_model.inference_and_postprocess(pre, batch)
            torch.cuda.synchronize()
            t = time.perf_counter()
            n = 0
            for _ in range(MEASURE_ITERS):
                det_model.inference_and_postprocess(pre, batch)
                n += 4
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            fps = n / dt
            print(f"  imgsz={imgsz:>3} (-> {det_model.imgsz}): {fps:6.1f} fps")
        except Exception as e:
            print(f"  imgsz={imgsz:>3}: FAILED -> {type(e).__name__}: {e}")
    det_model.imgsz = orig_imgsz


def main():
    rp, dp = resolve()
    device = torch.device(DEVICE)
    print(f"loading models... restoration={rp}\n                  detection={dp}")
    det_model, rest_model, pad_mode = load_models(
        device, RESTORATION_MODEL, rp, None, dp, fp16=FP16, detect_face_mosaics=False)
    print("loaded.")
    bench_vsr(rest_model, device)
    bench_yolo(det_model, device)


if __name__ == "__main__":
    main()
