# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Steady-state distribution of on-screen/sink present intervals from a realtime_trace run.

Reads present_*.csv (one monotonic_ns per shown frame), drops the first SKIP seconds (startup
outliers), and prints percentiles + a 2ms-bin histogram so we can see the real shape (e.g.
whether the 31/47 'bimodal' is genuinely two peaks). Usage:
    python scripts/analyze_present.py <trace_dir> [skip_seconds]
"""
import glob
import os
import statistics
import sys


def load_intervals(trace_dir, skip_s=5.0):
    files = sorted(glob.glob(os.path.join(trace_dir, "present_*.csv")))
    if not files:
        return None, None
    path = files[-1]
    ts = []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                try:
                    ts.append(int(line.split(",")[0]))
                except ValueError:
                    pass
    if len(ts) < 3:
        return path, []
    t0 = ts[0]
    cutoff = t0 + skip_s * 1e9
    ts = [t for t in ts if t >= cutoff]
    ivs = [(ts[i] - ts[i - 1]) / 1e6 for i in range(1, len(ts))]
    return path, ivs


def pctile(s, q):
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(pos); hi = min(lo + 1, len(s) - 1); frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def report(label, trace_dir, skip_s=5.0):
    path, ivs = load_intervals(trace_dir, skip_s)
    if path is None:
        print(f"{label}: no present csv in {trace_dir}")
        return
    if not ivs:
        print(f"{label}: too few events")
        return
    s = sorted(ivs)
    n = len(s)
    late40 = 100.0 * sum(1 for v in s if v > 40) / n
    late43 = 100.0 * sum(1 for v in s if v > 43) / n
    print(f"\n=== {label} ===  ({os.path.basename(path)}, {n} intervals, skip {skip_s:.0f}s)")
    print(f"  mean={statistics.fmean(s):.2f}  median={statistics.median(s):.2f}  stddev={statistics.pstdev(s):.2f}")
    print(f"  p10={pctile(s,.10):.2f} p25={pctile(s,.25):.2f} p50={pctile(s,.50):.2f} "
          f"p75={pctile(s,.75):.2f} p90={pctile(s,.90):.2f} p95={pctile(s,.95):.2f} p99={pctile(s,.99):.2f}")
    print(f"  %>40ms={late40:.1f}  %>43ms={late43:.1f}  max={s[-1]:.1f}")
    # 2ms-bin histogram from 24 to 52 ms
    bins = {}
    for v in s:
        b = int(v // 2) * 2
        bins[b] = bins.get(b, 0) + 1
    print("  hist(2ms bins):")
    for b in sorted(bins):
        if 20 <= b <= 60:
            bar = "#" * max(1, int(60 * bins[b] / n))
            print(f"    {b:>3}-{b+2:<3}ms {bins[b]:>5} {100*bins[b]/n:4.1f}% {bar}")


if __name__ == "__main__":
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    report(sys.argv[1], sys.argv[1], skip)
