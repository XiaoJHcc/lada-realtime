# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Simulate a PyInstaller module `excludes=[...]` and run the full demosaic load
+ inference chain. Installs a meta-path finder that makes the named top-level
packages unimportable (raising ModuleNotFoundError exactly as a frozen build
without them would), then runs scripts/dll_trim_verify.py. If any module in the
YOLO/BasicVSR++ load path top-level-imports an excluded package, this FAILS —
which is the signal not to exclude it. Not part of the shipped app.

Usage: exclude_sim_verify.py <comma-separated-modules> [video]"""
import runpy
import sys

BLOCK = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set()
VIDEO = sys.argv[2] if len(sys.argv) > 2 else "test_video.mp4"


class _Blocker:
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCK:
            raise ModuleNotFoundError(f"simulated-exclude: {name}")
        return None


sys.meta_path.insert(0, _Blocker())
print(f"[exclude-sim] blocking imports of: {sorted(BLOCK)}")

# sanity: confirm the blocker actually bites
for m in sorted(BLOCK):
    try:
        __import__(m)
        print(f"[exclude-sim] WARNING: {m} imported despite block")
    except ModuleNotFoundError:
        print(f"[exclude-sim] confirmed {m} is unimportable")

sys.argv = ["dll_trim_verify", VIDEO]
runpy.run_path("scripts/dll_trim_verify.py", run_name="__main__")
