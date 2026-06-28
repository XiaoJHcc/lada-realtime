# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""
Per-frame realtime playback tracer (diagnostics only).

Goal: answer "is the displayed frame rate low / unevenly distributed, and if so where
is the loss?" by recording two independent event streams during realtime playback and
dumping them to CSV + a console summary.

- PUSH events: recorded in the appsrc worker each time a buffer is pushed. Tells us the
  PRODUCTION cadence (can the worker sustain ~fps under GIL/GPU contention) and whether
  per-frame processing (CPU copy / tobytes / push) spikes.
- SINK events: recorded by a buffer pad probe on the gtk4paintablesink sink pad. Since the
  sink syncs each buffer to the pipeline clock, the inter-arrival cadence at that pad is the
  DISPLAY cadence. Comparing the clock running-time at arrival against the buffer PTS tells
  us whether frames arrive on time or late (late => juddery / QoS-dropped).
- SINK STATS samples: GstBaseSink's `stats` (rendered / dropped / average-rate), polled
  periodically, gives an authoritative dropped-frame count.

Everything here is OFF unless the env var LADA_REALTIME_TRACE is set (truthy). When off,
TRACE_ENABLED is False and the call sites are a single cheap bool check -> zero overhead on
the normal path. The value of LADA_REALTIME_TRACE may be an output directory; otherwise the
trace is written to <project_root>/realtime_trace/.

The hot-path record_*() calls only append to a collections.deque (GIL-atomic, no lock) and
never raise into the caller, so they can't stall the push loop or the streaming thread.
"""

import logging
import os
import statistics
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


_TRACE_ENV = os.environ.get("LADA_REALTIME_TRACE")
TRACE_ENABLED: bool = _env_truthy(_TRACE_ENV)


def _default_output_dir() -> str:
    # If the env var holds a path (anything that isn't a bare truthy flag), use it as the
    # output dir; else default to <project_root>/realtime_trace/. project_root = repo root,
    # i.e. three parents up from this file (lada/gui/realtime/realtime_trace.py).
    val = (_TRACE_ENV or "").strip()
    if val and val.lower() not in ("1", "true", "yes", "on"):
        return val
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(project_root, "realtime_trace")


class RealtimeTracer:
    """Thread-safe collector for realtime playback per-frame timing.

    A single instance is shared across the appsrc (push events), the pipeline manager (sink
    pad probe + sink stats) and the view (dump on close/EOS). Access via get_tracer()."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # deques: append from hot paths without a lock (CPython list/deque append is atomic
        # under the GIL). We snapshot-by-copy under _lock only in dump()/summary().
        self._push = deque()   # (offset, pts_ns, push_monotonic_ns, wait_ns, process_ns, hit)
        self._sink = deque()   # (pts_ns, arrival_monotonic_ns, running_time_ns)
        self._sinkstats = deque()  # (monotonic_ns, rendered, dropped, average_rate, qlevel_ns, qbuffers)
        # Presentation-layer ground truth (GTK main thread):
        self._present = deque()  # (monotonic_ns,) timestamp each new frame the paintable shows
        self._ticks = deque()    # (frame_time_us, newframe_count) one per GdkFrameClock tick (screen refresh)
        self._newframe_count = 0
        self._lock = threading.Lock()
        self._fps_target = 0.0
        # de-dupe the run id (timestamp) so periodic rewrites overwrite the same files
        self._run_id = time.strftime("%Y%m%d_%H%M%S")
        self._last_summary_log = 0.0
        os.makedirs(self.output_dir, exist_ok=True)
        logger.warning(f"realtime trace ENABLED -> writing to {self.output_dir} (run {self._run_id})")

    def set_fps_target(self, fps: float):
        if fps and fps > 0:
            self._fps_target = float(fps)

    # ---- hot-path recorders (must never raise) -------------------------------------------
    def record_push(self, offset: int, pts_ns: int, push_monotonic_ns: int,
                    wait_ns: int, process_ns: int, hit: bool):
        try:
            self._push.append((offset, pts_ns, push_monotonic_ns, wait_ns, process_ns, 1 if hit else 0))
        except Exception:
            pass

    def record_sink(self, pts_ns: int, arrival_monotonic_ns: int, running_time_ns: int):
        try:
            self._sink.append((pts_ns, arrival_monotonic_ns, running_time_ns))
        except Exception:
            pass

    def record_sink_stats(self, monotonic_ns: int, rendered: int, dropped: int, average_rate: float,
                          qlevel_ns: int = -1, qbuffers: int = -1):
        try:
            self._sinkstats.append((monotonic_ns, rendered, dropped, average_rate, qlevel_ns, qbuffers))
        except Exception:
            pass

    def record_newframe(self, monotonic_ns: int):
        """A new frame became visible (paintable invalidate-contents). The true presentation
        cadence -- even spacing here == smooth playback regardless of the GStreamer side."""
        try:
            self._newframe_count += 1
            self._present.append((monotonic_ns,))
        except Exception:
            pass

    def record_tick(self, frame_time_us: int):
        """One GdkFrameClock tick (a screen-refresh opportunity). Pairs the tick time with the
        running new-frame count so we can see how many refreshes each frame is held for."""
        try:
            self._ticks.append((frame_time_us, self._newframe_count))
        except Exception:
            pass

    def reset(self):
        """Drop accumulated events (e.g. on a fresh file open) but keep the same run id."""
        with self._lock:
            self._push.clear()
            self._sink.clear()
            self._sinkstats.clear()
            self._present.clear()
            self._ticks.clear()

    # ---- analysis / output ---------------------------------------------------------------
    @staticmethod
    def _intervals_ms(monotonic_ns_list: list[int]) -> list[float]:
        out = []
        for i in range(1, len(monotonic_ns_list)):
            dt = (monotonic_ns_list[i] - monotonic_ns_list[i - 1]) / 1e6
            if dt >= 0:
                out.append(dt)
        return out

    @staticmethod
    def _pctile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = q * (len(sorted_vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    def _interval_stats(self, intervals_ms: list[float]) -> dict:
        if not intervals_ms:
            return {"n": 0}
        s = sorted(intervals_ms)
        return {
            "n": len(s),
            "mean": statistics.fmean(s),
            "median": statistics.median(s),
            "stddev": statistics.pstdev(s) if len(s) > 1 else 0.0,
            "p95": self._pctile(s, 0.95),
            "p99": self._pctile(s, 0.99),
            "min": s[0],
            "max": s[-1],
        }

    def _build_summary(self) -> tuple[str, dict]:
        with self._lock:
            push = list(self._push)
            sink = list(self._sink)
            sinkstats = list(self._sinkstats)
            present = list(self._present)
            ticks = list(self._ticks)

        target = self._fps_target or 0.0
        target_interval_ms = (1000.0 / target) if target > 0 else 0.0

        lines = []
        lines.append("=== realtime playback trace summary ===")
        lines.append(f"run id: {self._run_id}")
        lines.append(f"fps target: {target:.3f}" + (f"  (frame interval {target_interval_ms:.2f} ms)" if target > 0 else ""))
        lines.append("")

        info = {"target_fps": target}

        # --- PUSH side (production cadence) ---
        if push:
            push_mono = [p[2] for p in push]
            span_s = (push_mono[-1] - push_mono[0]) / 1e9 if len(push_mono) > 1 else 0.0
            eff_push_fps = (len(push) - 1) / span_s if span_s > 0 else 0.0
            push_iv = self._interval_stats(self._intervals_ms(push_mono))
            proc_ms = sorted(p[4] / 1e6 for p in push)
            wait_ms = sorted(p[3] / 1e6 for p in push)
            hits = sum(p[5] for p in push)
            hit_rate = hits / len(push) if push else 0.0
            info["push"] = {"count": len(push), "eff_fps": eff_push_fps, "intervals": push_iv,
                            "hit_rate": hit_rate}
            lines.append(f"[PUSH] frames={len(push)}  span={span_s:.2f}s  effective push fps={eff_push_fps:.2f}")
            lines.append(f"[PUSH] AI hit rate={hit_rate*100:.1f}%  ({hits}/{len(push)})")
            if push_iv["n"]:
                lines.append(f"[PUSH] push interval ms: mean={push_iv['mean']:.2f} median={push_iv['median']:.2f} "
                             f"stddev={push_iv['stddev']:.2f} p95={push_iv['p95']:.2f} p99={push_iv['p99']:.2f} max={push_iv['max']:.2f}")
            if proc_ms:
                lines.append(f"[PUSH] per-frame process ms: mean={statistics.fmean(proc_ms):.2f} median={statistics.median(proc_ms):.2f} "
                             f"p95={self._pctile(proc_ms, 0.95):.2f} p99={self._pctile(proc_ms, 0.99):.2f} max={proc_ms[-1]:.2f}")
                lines.append(f"[PUSH] passthrough get() wait ms: mean={statistics.fmean(wait_ms):.2f} median={statistics.median(wait_ms):.2f} max={wait_ms[-1]:.2f}")
        else:
            lines.append("[PUSH] no events")

        lines.append("")

        # --- SINK side (display cadence) ---
        if sink:
            sink_mono = [s[1] for s in sink]
            span_s = (sink_mono[-1] - sink_mono[0]) / 1e9 if len(sink_mono) > 1 else 0.0
            eff_sink_fps = (len(sink) - 1) / span_s if span_s > 0 else 0.0
            sink_iv = self._interval_stats(self._intervals_ms(sink_mono))
            # lateness: running_time - pts (positive => arrived after its display deadline).
            # Valid only for straight playback (segment starts at 0 so running_time ~= pts).
            lateness_ms = sorted((s[2] - s[0]) / 1e6 for s in sink)
            half_frame_ms = target_interval_ms / 2.0 if target_interval_ms > 0 else 0.0
            late_count = sum(1 for s in sink if half_frame_ms > 0 and (s[2] - s[0]) / 1e6 > half_frame_ms)
            info["sink"] = {"count": len(sink), "eff_fps": eff_sink_fps, "intervals": sink_iv,
                            "late_count": late_count}
            lines.append(f"[SINK] frames={len(sink)}  span={span_s:.2f}s  effective display fps={eff_sink_fps:.2f}")
            if sink_iv["n"]:
                lines.append(f"[SINK] display interval ms: mean={sink_iv['mean']:.2f} median={sink_iv['median']:.2f} "
                             f"stddev={sink_iv['stddev']:.2f} p95={sink_iv['p95']:.2f} p99={sink_iv['p99']:.2f} max={sink_iv['max']:.2f}")
                # how many display gaps exceed 1.5x the target frame interval (a visible hitch)
                if target_interval_ms > 0:
                    ivs = self._intervals_ms(sink_mono)
                    big = sum(1 for v in ivs if v > 1.5 * target_interval_ms)
                    huge = sum(1 for v in ivs if v > 2.5 * target_interval_ms)
                    lines.append(f"[SINK] gaps >1.5x target={big}  >2.5x target={huge}  (of {len(ivs)} intervals)")
            if half_frame_ms > 0:
                lines.append(f"[SINK] late arrivals (running_time > pts+half_frame)={late_count}/{len(sink)} "
                             f"| lateness ms: median={statistics.median(lateness_ms):.2f} p95={self._pctile(lateness_ms, 0.95):.2f} max={lateness_ms[-1]:.2f}")
                lines.append("       (lateness valid for straight playback only; ignore if you seeked)")
        else:
            lines.append("[SINK] no events (pad probe not firing?)")

        lines.append("")

        # --- SINK STATS (authoritative rendered/dropped) + buffer depth ---
        if sinkstats:
            first = sinkstats[0]
            last = sinkstats[-1]
            rendered = last[1] - first[1]
            dropped = last[2] - first[2]
            total = rendered + dropped
            drop_pct = (dropped / total * 100.0) if total > 0 else 0.0
            info["sinkstats"] = {"rendered": rendered, "dropped": dropped, "drop_pct": drop_pct,
                                 "average_rate": last[3]}
            lines.append(f"[SINK STATS] rendered={rendered}  dropped={dropped}  drop%={drop_pct:.2f}  "
                         f"avg-rate(last)={last[3]:.3f}")
            # Buffer depth: how many frames the jitter queue holds ahead of the sink. A normal
            # player keeps a deep buffer; if this is ~0-1 frames the sink has no slack to absorb
            # delivery jitter -> uneven release. qlevel_ns/qbuffers added later in the tuple.
            if len(last) > 4:
                qlevels_ms = sorted(s[4] / 1e6 for s in sinkstats if len(s) > 4 and s[4] >= 0)
                qbufs = sorted(s[5] for s in sinkstats if len(s) > 5 and s[5] >= 0)
                if qlevels_ms:
                    fpf = target_interval_ms or 33.37
                    lines.append(f"[BUFFER] jitter-queue level ms: median={statistics.median(qlevels_ms):.1f} "
                                 f"min={qlevels_ms[0]:.1f} max={qlevels_ms[-1]:.1f}  "
                                 f"(~{statistics.median(qlevels_ms)/fpf:.1f} frames median ahead)")
                if qbufs:
                    lines.append(f"[BUFFER] jitter-queue buffers: median={statistics.median(qbufs)} min={qbufs[0]} max={qbufs[-1]}")
        else:
            lines.append("[SINK STATS] unavailable")

        lines.append("")

        # --- PRESENT (true on-screen cadence: paintable new-frame events, GTK main thread) ---
        if present:
            pres_mono = [p[0] for p in present]
            span_s = (pres_mono[-1] - pres_mono[0]) / 1e9 if len(pres_mono) > 1 else 0.0
            eff_pres_fps = (len(pres_mono) - 1) / span_s if span_s > 0 else 0.0
            pres_iv = self._interval_stats(self._intervals_ms(pres_mono))
            info["present"] = {"count": len(present), "eff_fps": eff_pres_fps, "intervals": pres_iv}
            lines.append(f"[PRESENT] new frames shown={len(present)}  span={span_s:.2f}s  effective on-screen fps={eff_pres_fps:.2f}")
            if pres_iv["n"]:
                lines.append(f"[PRESENT] frame-on-screen interval ms: mean={pres_iv['mean']:.2f} median={pres_iv['median']:.2f} "
                             f"stddev={pres_iv['stddev']:.2f} p95={pres_iv['p95']:.2f} p99={pres_iv['p99']:.2f} max={pres_iv['max']:.2f}")
                if target_interval_ms > 0:
                    ivs = self._intervals_ms(pres_mono)
                    big = sum(1 for v in ivs if v > 1.5 * target_interval_ms)
                    huge = sum(1 for v in ivs if v > 2.5 * target_interval_ms)
                    lines.append(f"[PRESENT] gaps >1.5x target={big}  >2.5x target={huge}  (of {len(ivs)} intervals)")
        else:
            lines.append("[PRESENT] no events (paintable invalidate-contents not wired / widget unmapped)")

        # --- TICKS (GdkFrameClock: refresh cadence + refreshes-held-per-frame) ---
        if ticks and len(ticks) > 2:
            tick_us = [t[0] for t in ticks]
            tiv = sorted((tick_us[i] - tick_us[i - 1]) / 1000.0 for i in range(1, len(tick_us)))
            refresh_ms = statistics.median(tiv) if tiv else 0.0
            refresh_hz = (1000.0 / refresh_ms) if refresh_ms > 0 else 0.0
            # refreshes each frame is held for = ticks between consecutive new-frame increments
            counts = [t[1] for t in ticks]
            held = []
            run = 0
            for i in range(1, len(counts)):
                run += 1
                if counts[i] != counts[i - 1]:
                    held.append(run)
                    run = 0
            from collections import Counter as _C
            held_hist = dict(sorted(_C(held).items()))
            lines.append(f"[TICKS] frame-clock refresh: median interval={refresh_ms:.2f} ms (~{refresh_hz:.0f} Hz), ticks={len(ticks)}")
            if held:
                lines.append(f"[TICKS] refreshes held per shown frame: median={statistics.median(held):.1f} "
                             f"distribution(refreshes:count)={held_hist}")
                lines.append("        (even playback = one dominant value; spread = judder. e.g. 30fps@150Hz should be a clean 5)")
        else:
            lines.append("[TICKS] no events (tick callback not wired / widget unmapped)")

        lines.append("=== end summary ===")
        return "\n".join(lines), info

    def maybe_log_rolling_summary(self, min_interval_s: float = 5.0):
        """Called from a periodic timeout. Logs a summary + rewrites CSVs at most every
        min_interval_s so even an unclean exit leaves data on disk."""
        now = time.monotonic()
        if now - self._last_summary_log < min_interval_s:
            return
        self._last_summary_log = now
        try:
            self.dump(log_summary=True)
        except Exception as e:
            logger.warning(f"realtime trace rolling dump failed: {e}")

    def dump(self, log_summary: bool = True):
        """Write CSVs + summary.txt and optionally log the summary to the console."""
        with self._lock:
            push = list(self._push)
            sink = list(self._sink)
            sinkstats = list(self._sinkstats)
            present = list(self._present)
            ticks = list(self._ticks)

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"push_{self._run_id}.csv"), "w", encoding="utf-8") as f:
                f.write("offset,pts_ns,push_monotonic_ns,wait_ns,process_ns,hit\n")
                for r in push:
                    f.write(",".join(str(x) for x in r) + "\n")
            with open(os.path.join(self.output_dir, f"sink_{self._run_id}.csv"), "w", encoding="utf-8") as f:
                f.write("pts_ns,arrival_monotonic_ns,running_time_ns\n")
                for r in sink:
                    f.write(",".join(str(x) for x in r) + "\n")
            with open(os.path.join(self.output_dir, f"sinkstats_{self._run_id}.csv"), "w", encoding="utf-8") as f:
                f.write("monotonic_ns,rendered,dropped,average_rate,qlevel_ns,qbuffers\n")
                for r in sinkstats:
                    f.write(",".join(str(x) for x in r) + "\n")
            with open(os.path.join(self.output_dir, f"present_{self._run_id}.csv"), "w", encoding="utf-8") as f:
                f.write("monotonic_ns\n")
                for r in present:
                    f.write(",".join(str(x) for x in r) + "\n")
            with open(os.path.join(self.output_dir, f"ticks_{self._run_id}.csv"), "w", encoding="utf-8") as f:
                f.write("frame_time_us,newframe_count\n")
                for r in ticks:
                    f.write(",".join(str(x) for x in r) + "\n")
        except Exception as e:
            logger.warning(f"realtime trace CSV write failed: {e}")

        summary, _info = self._build_summary()
        try:
            with open(os.path.join(self.output_dir, f"summary_{self._run_id}.txt"), "w", encoding="utf-8") as f:
                f.write(summary + "\n")
        except Exception as e:
            logger.warning(f"realtime trace summary write failed: {e}")

        if log_summary:
            logger.warning("\n" + summary)


_tracer: RealtimeTracer | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> RealtimeTracer | None:
    """Return the shared tracer when tracing is enabled, else None. Lazily constructed."""
    if not TRACE_ENABLED:
        return None
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = RealtimeTracer(_default_output_dir())
    return _tracer
