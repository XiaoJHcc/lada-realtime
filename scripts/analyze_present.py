# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Time-windowed distribution of realtime_trace timing streams.

Two independent streams are analysed over the SAME run so we can separate the cold-start
regime (heavy AI restart burst) from the steady-state regime (warm pipeline):
  - present_*.csv : on-screen cadence (paintable invalidate-contents, GTK main thread).
    Even spacing here == smooth playback. This is the ground-truth "what the eye sees".
  - push_*.csv    : production cadence in the appsrc worker (push interval) plus per-frame
    process_ns (CPU copy/tobytes/push) and wait_ns (blocked on passthrough get()). Spikes
    here during cold-start localise judder to the PRODUCTION side (GIL/GPU contention).

Usage:
    python scripts/analyze_present.py <trace_dir>                  # auto: cold-start vs steady
    python scripts/analyze_present.py <trace_dir> <skip_s>         # single present window (legacy)
    python scripts/analyze_present.py <trace_dir> <start_s> <end_s>  # explicit present window
"""
import glob
import os
import statistics
import sys


def _load_col(trace_dir, prefix, col=0):
    files = sorted(glob.glob(os.path.join(trace_dir, f"{prefix}_*.csv")))
    if not files:
        return None, []
    path = files[-1]
    vals = []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                vals.append(int(parts[col]))
            except (ValueError, IndexError):
                pass
    return path, vals


def _window(ts, start_s, end_s):
    """Filter monotonic-ns timestamps to [t0+start_s, t0+end_s)."""
    if not ts:
        return []
    t0 = ts[0]
    lo = t0 + start_s * 1e9
    hi = t0 + end_s * 1e9 if end_s is not None else float("inf")
    return [t for t in ts if lo <= t < hi]


def _intervals_ms(ts):
    return [(ts[i] - ts[i - 1]) / 1e6 for i in range(1, len(ts))]


def pctile(s, q):
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(pos); hi = min(lo + 1, len(s) - 1); frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats_line(ivs):
    s = sorted(ivs)
    n = len(s)
    if n == 0:
        return "  (no intervals)"
    late40 = 100.0 * sum(1 for v in s if v > 40) / n
    late43 = 100.0 * sum(1 for v in s if v > 43) / n
    big = sum(1 for v in s if v > 50)  # >1.5x of 33.4ms
    return (f"  n={n} mean={statistics.fmean(s):.2f} median={statistics.median(s):.2f} "
            f"stddev={statistics.pstdev(s):.2f}\n"
            f"  p50={pctile(s,.50):.2f} p90={pctile(s,.90):.2f} p95={pctile(s,.95):.2f} "
            f"p99={pctile(s,.99):.2f} max={s[-1]:.1f}\n"
            f"  %>40ms={late40:.1f} %>43ms={late43:.1f} gaps>50ms={big}")


def _hist(ivs, lo=20, hi=60):
    s = sorted(ivs)
    n = len(s)
    if not n:
        return
    bins = {}
    for v in s:
        b = int(v // 2) * 2
        bins[b] = bins.get(b, 0) + 1
    for b in sorted(bins):
        if lo <= b <= hi:
            bar = "#" * max(1, int(60 * bins[b] / n))
            print(f"    {b:>3}-{b+2:<3}ms {bins[b]:>5} {100*bins[b]/n:4.1f}% {bar}")


def present_report(trace_dir, start_s, end_s, show_hist=True):
    path, ts = _load_col(trace_dir, "present", 0)
    if path is None:
        print(f"  PRESENT: no present_*.csv in {trace_dir}")
        return
    win = _window(ts, start_s, end_s)
    ivs = _intervals_ms(win)
    lbl = f"[{start_s:.0f}s-{'end' if end_s is None else f'{end_s:.0f}s'}]"
    print(f"  PRESENT (on-screen) {lbl}:")
    print(_stats_line(ivs))
    if show_hist and ivs:
        _hist(ivs)


def push_report(trace_dir, start_s, end_s):
    # push_*.csv columns: offset,pts_ns,push_monotonic_ns,wait_ns,process_ns,hit
    files = sorted(glob.glob(os.path.join(trace_dir, "push_*.csv")))
    if not files:
        print("  PUSH: no push_*.csv")
        return
    rows = []
    with open(files[-1], encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 6:
                try:
                    rows.append((int(p[2]), int(p[3]), int(p[4]), int(p[5])))  # mono, wait, proc, hit
                except ValueError:
                    pass
    if not rows:
        print("  PUSH: empty")
        return
    t0 = rows[0][0]
    lo = t0 + start_s * 1e9
    hi = t0 + end_s * 1e9 if end_s is not None else float("inf")
    win = [r for r in rows if lo <= r[0] < hi]
    if len(win) < 2:
        print("  PUSH: too few in window")
        return
    push_mono = [r[0] for r in win]
    push_ivs = _intervals_ms(push_mono)
    proc_ms = sorted(r[2] / 1e6 for r in win)
    wait_ms = sorted(r[1] / 1e6 for r in win)
    hits = sum(r[3] for r in win)
    lbl = f"[{start_s:.0f}s-{'end' if end_s is None else f'{end_s:.0f}s'}]"
    print(f"  PUSH (production) {lbl}:")
    print(f"    push interval: median={statistics.median(push_ivs):.2f} stddev={statistics.pstdev(push_ivs):.2f} "
          f"p95={pctile(sorted(push_ivs),.95):.2f} p99={pctile(sorted(push_ivs),.99):.2f} max={max(push_ivs):.1f}")
    print(f"    process_ns ms: median={statistics.median(proc_ms):.2f} p95={pctile(proc_ms,.95):.2f} "
          f"p99={pctile(proc_ms,.99):.2f} max={proc_ms[-1]:.1f}")
    print(f"    wait_ns(get) ms: median={statistics.median(wait_ms):.2f} p95={pctile(wait_ms,.95):.2f} max={wait_ms[-1]:.1f}")
    print(f"    AI hit rate: {100*hits/len(win):.1f}% ({hits}/{len(win)})")


if __name__ == "__main__":
    trace_dir = sys.argv[1]
    if len(sys.argv) == 2:
        # auto mode: cold-start vs steady-state for both streams
        print(f"\n########## {trace_dir} ##########")
        print("\n=== COLD-START window [0s-6s] ===")
        present_report(trace_dir, 0.0, 6.0, show_hist=True)
        push_report(trace_dir, 0.0, 6.0)
        print("\n=== STEADY-STATE window [10s-end] ===")
        present_report(trace_dir, 10.0, None, show_hist=True)
        push_report(trace_dir, 10.0, None)
    elif len(sys.argv) == 3:
        present_report(trace_dir, float(sys.argv[2]), None)
        push_report(trace_dir, float(sys.argv[2]), None)
    else:
        present_report(trace_dir, float(sys.argv[2]), float(sys.argv[3]))
        push_report(trace_dir, float(sys.argv[2]), float(sys.argv[3]))
