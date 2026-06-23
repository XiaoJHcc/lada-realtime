# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""
Cold-start + steady-state profiler for the realtime path (v2).

v1 mistake: it never enabled the frontier gate, so YOLO ran unbounded and
stole ~30% of the restorer's GPU, reading 26fps where the real player shows
35-38. v2 fixes that by driving the SAME gate the realtime appsrc drives:

  - a simulated playhead thread advances at video fps and calls
    set_processing_frontier(playhead + window) every frame, exactly like
    RealtimeFrameRestorerAppSrc._update_processing_frontier
  - the internal FrameRestorer._detector_lead (default 2*clip) is overridden
    per-cell so we can A/B 2*clip (current) vs 1*clip (cost E fix)

Measures per (seek, clip_T, detector_lead) cell:
  - first restored frame wall time (cold-start first-frame latency)
  - detector frames done at first frame (how far ahead YOLO ran = cost E)
  - steady-state restorer fps after warmup region (should reproduce 35-38)
  - per-clip forward ms

Averages over several seek points (hard stretches differ across the file).

Run:
  .\.venv\Scripts\python.exe scripts\realtime_coldstart_profile.py
"""
import logging
import os
import threading
import time

os.environ.setdefault("LOG_LEVEL", "WARNING")

import torch

from lada.utils import video_utils
from lada.restorationpipeline import load_models
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, ErrorMarker
from lada import ModelFiles

logging.basicConfig(level=logging.WARNING)

VIDEO = "test_video.mp4"
DEVICE = "cuda"
FP16 = True
RESTORATION_MODEL = "basicvsrpp-generic-v1.2"
DETECTION_MODEL = "v4-fast"
SEEK_SECONDS_LIST = [10.0, 30.0, 50.0, 70.0, 90.0]
LOOKAHEAD_FRAMES = 180  # config default; window = max(lookahead, 2*clip)


def resolve_model_names():
    import glob
    try:
        rest_path = ModelFiles.get_restoration_model_by_name(RESTORATION_MODEL).path
    except Exception:
        c = glob.glob("model_weights/*restoration*v1.2*.pth"); rest_path = c[0] if c else None
    try:
        det_path = ModelFiles.get_detection_model_by_name(DETECTION_MODEL).path
    except Exception:
        c = glob.glob("model_weights/*detection*v4*fast*.pt"); det_path = c[0] if c else None
    return RESTORATION_MODEL, rest_path, DETECTION_MODEL, det_path


class PlayheadDriver(threading.Thread):
    """Advance a simulated playhead at video fps, driving the frontier gate exactly like
    RealtimeFrameRestorerAppSrc does (playhead + window), so the detector is gated and can't
    starve the restorer of GPU. window = max(lookahead, 2*clip), per appsrc."""
    def __init__(self, fr: FrameRestorer, start_frame: int, fps: float, clip_T: int):
        super().__init__(daemon=True)
        self.fr = fr
        self.start_frame = start_frame
        self.frame_dt = 1.0 / fps
        self.window = max(LOOKAHEAD_FRAMES, 2 * clip_T)
        self.playhead = start_frame
        self._stop = False

    def run(self):
        next_t = time.perf_counter()
        while not self._stop:
            self.fr.set_processing_frontier(self.playhead + self.window)
            self.playhead += 1
            next_t += self.frame_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    def stop(self):
        self._stop = True


def run_cell(seek_s, clip_T, detector_lead, det_model, rest_model, pad_mode, device, meta, fps,
             long_frames=0):
    start_ns = int(seek_s * 1e9)
    clip_proc_log = []
    _orig_restore = rest_model.restore
    def _timed_restore(video, *a, **kw):
        torch.cuda.synchronize(); _t = time.perf_counter()
        out = _orig_restore(video, *a, **kw)
        torch.cuda.synchronize(); clip_proc_log.append((len(video), time.perf_counter() - _t))
        return out
    rest_model.restore = _timed_restore
    try:
        fr = FrameRestorer(device, VIDEO, clip_T, RESTORATION_MODEL, det_model, rest_model,
                           pad_mode, mosaic_detection=False)
        # cost E A/B: override the internal just-in-time detector lead (default 2*clip).
        fr._detector_lead = max(1, detector_lead)

        t0 = time.perf_counter()
        fr.start(start_ns=start_ns)
        driver = PlayheadDriver(fr, fr.start_frame, fps, clip_T)
        driver.start()

        first_frame_wall = None
        det_at_first = rest_at_first = None
        frames_seen = 0
        out_q = fr.get_frame_restoration_queue()
        # short cold-start probe (5 clips) unless long_frames requested for steady-state.
        target = long_frames if long_frames else clip_T * 5
        # steady-state delivered throughput: measure wall time across a window of frames
        # AFTER the cold-start region (skip first 2 clips), so we capture real delivered
        # rate incl. any restorer stalls waiting on the detector — the thing forward-fps hides.
        steady_window_start_frame = clip_T * 2
        steady_t_start = None
        steady_frames = 0
        deadline = t0 + (90.0 if long_frames else 40.0)
        while time.perf_counter() < deadline:
            try:
                elem = out_q.get(timeout=8.0)
            except Exception:
                break
            if elem is EOF_MARKER or elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                break
            if first_frame_wall is None:
                first_frame_wall = time.perf_counter() - t0
                det_at_first = fr.get_detector_frames_done()
                rest_at_first = fr.get_restorer_frames_done()
            frames_seen += 1
            if frames_seen == steady_window_start_frame:
                steady_t_start = time.perf_counter()
            elif steady_t_start is not None:
                steady_frames += 1
            if frames_seen >= target:
                break

        steady_wall = (time.perf_counter() - steady_t_start) if steady_t_start else None
        # delivered fps = frames actually emitted per wall second in the steady window.
        # This INCLUDES restorer stalls (idle waiting for detector), unlike forward-fps.
        delivered_fps = (steady_frames / steady_wall) if (steady_wall and steady_wall > 0) else 0.0

        # forward-only fps (compute rate, excludes stalls) for comparison
        fwd = clip_proc_log[1:] if len(clip_proc_log) > 1 else clip_proc_log
        forward_fps = (sum(n for n, _ in fwd) / sum(p for _, p in fwd)) if fwd else 0.0
        result = dict(
            first_ms=(first_frame_wall * 1000) if first_frame_wall else None,
            det_at_first=det_at_first, rest_at_first=rest_at_first,
            forward_fps=forward_fps, delivered_fps=delivered_fps,
            steady_frames=steady_frames,
            clip0_ms=clip_proc_log[0][1] * 1000 if clip_proc_log else None,
            clip1_ms=clip_proc_log[1][1] * 1000 if len(clip_proc_log) > 1 else None,
        )
        driver.stop()
        fr.stop()
        return result
    finally:
        rest_model.restore = _orig_restore


def main():
    meta = video_utils.get_video_meta_data(VIDEO)
    fps = float(meta.video_fps)
    print(f"video: {meta.video_width}x{meta.video_height} @ {fps:.3f}fps, {meta.frames_count} frames")
    print(f"seek points: {SEEK_SECONDS_LIST}s, lookahead window={LOOKAHEAD_FRAMES}\n")

    rest_name, rest_path, det_name, det_path = resolve_model_names()
    device = torch.device(DEVICE)
    print("loading + warming models (once)...")
    det_model, rest_model, pad_mode = load_models(
        device, rest_name, rest_path, None, det_path, fp16=FP16, detect_face_mosaics=False)
    print("loaded.\n")

    # A/B matrix: clip_T x detector_lead. detector_lead in units of clip_T.
    # Cold-start probe (5 clips) first, then a LONG steady-state run (600 frames ~ 20 clips)
    # measuring DELIVERED throughput (incl. restorer stalls), which forward-fps hides.
    print("########## COLD-START PROBE (5 clips) ##########\n")
    for clip_T in (15, 30):
        for lead_mult in (2, 1.2, 1):
            lead = max(clip_T, int(round(lead_mult * clip_T)))
            firsts, dets = [], []
            print(f"===== clip_T={clip_T}, detector_lead={lead} ({lead_mult}*clip) =====")
            for seek_s in SEEK_SECONDS_LIST:
                r = run_cell(seek_s, clip_T, lead, det_model, rest_model, pad_mode, device, meta, fps)
                if r["first_ms"] is None:
                    print(f"  seek {seek_s:>5.0f}s: no restored frame (no mosaic in window?)")
                    continue
                firsts.append(r["first_ms"]); dets.append(r["det_at_first"])
                print(f"  seek {seek_s:>5.0f}s: first={r['first_ms']:.0f}ms  det@first={r['det_at_first']}")
                time.sleep(0.3)
            if firsts:
                print(f"  --- AVG first={sum(firsts)/len(firsts):.0f}ms  det@first={sum(dets)/len(dets):.0f} ---")
            print()

    print("\n########## LONG STEADY-STATE (600 frames, greedy consume) ##########")
    print("delivered_fps = frames emitted per WALL second incl. restorer stalls (the real metric).")
    print("forward_fps   = pure model compute rate (hides stalls). If 1*clip starves the")
    print("                restorer, delivered_fps drops vs 2*clip even if forward_fps rose.\n")
    LONG = 600
    for clip_T in (30,):
        for lead_mult in (2, 1.2, 1):
            lead = max(clip_T, int(round(lead_mult * clip_T)))
            delivs, fwds = [], []
            print(f"===== clip_T={clip_T}, detector_lead={lead} ({lead_mult}*clip) =====")
            for seek_s in SEEK_SECONDS_LIST:
                r = run_cell(seek_s, clip_T, lead, det_model, rest_model, pad_mode, device, meta, fps,
                             long_frames=LONG)
                if r["first_ms"] is None:
                    print(f"  seek {seek_s:>5.0f}s: no restored frame")
                    continue
                delivs.append(r["delivered_fps"]); fwds.append(r["forward_fps"])
                print(f"  seek {seek_s:>5.0f}s: delivered={r['delivered_fps']:.1f}fps  "
                      f"forward={r['forward_fps']:.1f}fps  (over {r['steady_frames']} frames)")
                time.sleep(0.3)
            if delivs:
                print(f"  --- AVG delivered={sum(delivs)/len(delivs):.1f}fps  "
                      f"forward={sum(fwds)/len(fwds):.1f}fps ---")
            print()


if __name__ == "__main__":
    main()
