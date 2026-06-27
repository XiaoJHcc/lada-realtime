# Windows Packaging Guide

End-to-end reference for building a distributable Windows release of **Lada Realtime**
(`lada.exe` + `lada-cli.exe`), including the gvsbuild GTK build, TensorRT bundling, and
the mirror/proxy configuration this repo ships with.

For *running from source* (not packaging), see [windows_install.md](windows_install.md)
and the "构建与运行" section of [CLAUDE.md](../CLAUDE.md).

---

## TL;DR

```powershell
# Full build (first time — installs system deps, builds GTK, downloads weights, packages)
.\packaging\windows\package_executable.ps1 -extra nvidia

# Rebuild the exe only, reusing an already-built GTK and system deps
.\packaging\windows\package_executable.ps1 -extra nvidia -skipWinget -skipGvsbuild
```

The script [`packaging/windows/package_executable.ps1`](../packaging/windows/package_executable.ps1)
is fully automated. Output lands in `dist/lada/` and (unless `-skipArchive`) a split 7z archive.

There is also a VS Code task — **"Package: build distributable .exe"** in
[.vscode/tasks.json](../.vscode/tasks.json) — that wraps the script with pickable flags.

### Script flags

| Flag | Effect |
|---|---|
| `-extra nvidia` \| `intel` | Target GPU variant (matches `pyproject.toml` extras). Default `nvidia`. |
| `-skipWinget` | Skip installing/upgrading system deps via winget. |
| `-skipGvsbuild` | Skip the gvsbuild GTK/GStreamer compile (reuse `build_gtk_release/`). |
| `-skipTranslations` | Don't compile `.po` → `.mo` (release ships without translations). |
| `-skipArchive` | Don't create the 7z archive (leaves `dist/lada/`). |
| `-cleanGvsbuild` | Clean rebuild of gvsbuild (use after upgrading gvsbuild/uv/python). |
| `-cliOnly` | Build only `lada-cli.exe` (skips the GTK build entirely). |

---

## Pipeline stages

The script runs these in order (each roughly idempotent — re-running with `-skip*` resumes):

1. **`Install-SystemDependencies`** (winget): FFmpeg, Git, uv, 7zip; for GUI also MSYS2,
   VS Build Tools, Rustup, VCRedist.
2. **`Build-SystemDependencies`** (gvsbuild): compiles GTK4, libadwaita, GStreamer + plugins
   into `build_gtk_release/`. Slow (tens of minutes to hours). See [gvsbuild pitfalls](#gvsbuild-pitfalls).
3. **`Compile-Translations`**: `.po` → `.mo` using the freshly built `msgfmt.exe`.
4. **`Download-ModelWeights`**: detection (`v4_fast`) + restoration (`generic_v1.2`)
   into `model_weights/`.
5. **`Install-PythonDependencies`**: creates `venv_release_win`, runs
   `uv sync --frozen --extra nvidia`, swaps `polars` → `polars-lts-cpu`, installs the
   project + pygobject/pycairo wheels + PyInstaller, applies the `patches/`.
6. **`Create-EXE`**: runs PyInstaller against [`lada.spec`](../packaging/windows/lada.spec).
7. **`Create-7ZArchive`**: splits into ≤2GB chunks for GitHub Releases.

> [!IMPORTANT]
> `uv sync` uses `--frozen`, so it installs **exactly** what's in `uv.lock`. If you add a
> dependency, run `uv lock` first or the release venv won't get it. This is precisely how
> the TensorRT packages were missing originally (see below).

---

## TensorRT bundling

This fork accelerates BasicVSR++ with TensorRT. Getting TRT into a *distributable* build
took two fixes that are easy to miss — both are now in place, documented here so they don't
regress.

### Why it broke (`No module named 'tensorrt'`)

The distributable would fail at "Build acceleration engines" with `No module named
'tensorrt'`. Two independent root causes:

1. **The dependency was never declared.** `torch-tensorrt` / `tensorrt` were originally
   `uv pip install`-ed into the dev `.venv` by hand, never written into `pyproject.toml` or
   `uv.lock`. Since packaging uses `uv sync --frozen`, the release venv never got them.
2. **PyInstaller couldn't see them.** These packages ship **no PyInstaller hooks**, and the
   code lazy-imports them (the TRT path falls back to PyTorch on any failure), so static
   analysis misses them and the native DLLs are never bundled.

### The fix (already applied)

**`pyproject.toml`** — TRT declared in the `nvidia` extra, pinned to the right indexes:

```toml
nvidia = [
    "torch==2.8.0; ...", "torchvision==0.23.0; ...",
    "torch-tensorrt==2.8.0; sys_platform == 'linux' or sys_platform == 'win32'",
    "tensorrt-cu12-libs==10.12.0.36; ...",       # explicit: transitive source pin is ignored otherwise
    "tensorrt-cu12-bindings==10.12.0.36; ...",
]
```

```toml
[tool.uv.sources]
torch-tensorrt          = [{ index = "pytorch-cu128", extra = "nvidia" }]
tensorrt-cu12-libs      = [{ index = "nvidia", extra = "nvidia" }]   # NVIDIA index, NOT the mirror
tensorrt-cu12-bindings  = [{ index = "nvidia", extra = "nvidia" }]
```

> **Why list the libs explicitly?** `tensorrt-cu12-libs` is a *transitive* dep of
> torch-tensorrt. uv silently ignores a `[tool.uv.sources]` index pin on a purely transitive
> package — it must be a **direct** dependency for the pin to apply. Without the pin, uv
> resolves the libs from the Tsinghua mirror, which only has a **709-byte sdist stub** (no
> real wheel) → install produces no DLLs.

**`pyproject.toml` → `[tool.uv]`** — pin resolution environments so uv doesn't choke on an
impossible platform split:

```toml
[tool.uv]
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'win32' and platform_machine == 'AMD64'",   # NB: Windows is 'AMD64', not 'x86_64'
]
```

> Without this, adding torch-tensorrt makes `uv lock` unsatisfiable: it tries to solve for
> aarch64+tegra (Jetson), where torch-tensorrt's tegra variant pulls `numpy<2`, clashing with
> opencv's `numpy>=2`. Lada doesn't ship for Jetson, so we constrain to real targets.
> **Don't add `darwin`** — it surfaces an `onnxruntime-gpu` (dev-group) conflict with no macOS
> wheel, and we don't ship for macOS.

**`packaging/windows/lada.spec`** — `collect_all` the four top-level packages and feed them
into both the GUI and CLI `Analysis` blocks:

```python
def get_trt_collection():
    trt_datas, trt_binaries, trt_hiddenimports = [], [], []
    for pkg in ("tensorrt", "tensorrt_libs", "tensorrt_bindings", "torch_tensorrt"):
        datas, binaries, hiddenimports = collect_all(pkg)
        ...
    return trt_datas, trt_binaries, trt_hiddenimports
```

Only collected for `-extra nvidia` (intel/cpu skip it). The CLI needs it too, for
`lada-cli --build-trt-engines`.

### Size cost

`tensorrt_libs` contains `nvinfer_builder_resource_10.dll` (~1.79GB) +
`nvinfer_10.dll` (~472MB). The distributable grows by **~2.2GB**. The builder resource is
only needed to *compile* engines — but since this fork's design is on-device compilation on
first run, it can't be dropped.

### Why the bundle's DLL loading works (no extra runtime hook needed)

`tensorrt_bindings/__init__.py` does `import tensorrt_libs` first; on success it sets
`_libs_wheel_imported = True` and **skips** the fragile `os.environ["PATH"]`-search fallback.
`tensorrt_libs/__init__.py` loads its DLLs via `CURDIR`-relative `ctypes.CDLL`, and
`collect_all` preserves the `tensorrt_libs/` directory layout — so the chain is self-contained
inside `_internal/`. torch_tensorrt's PATH-search fallback only triggers if `import tensorrt`
fails, which won't happen in a correct bundle.

### Engines are NOT shipped

TRT engines are bound to a specific GPU arch + TensorRT version and can't be distributed
across machines. End users compile them on first run:

- **GUI**: a first-run dialog ([`lada/gui/trt_setup_dialog.py`](../lada/gui/trt_setup_dialog.py))
  offers Build now / Later / pick-GPU.
- **CLI**: `lada-cli --build-trt-engines` prewarms them at install time.

Engines cache under `model_weights/<model>_sub_engines/` with arch / TRT-version / precision
encoded in the filename, so upgrading `torch-tensorrt` or swapping GPUs auto-invalidates them.
`LADA_BASICVSRPP_TRT=0` forces the PyTorch path.

---

## Mirrors & proxy

This repo's `pyproject.toml` defaults to **China-based mirrors** for speed inside mainland
China:

| Index | URL | Notes |
|---|---|---|
| PyPI (default) | `https://pypi.tuna.tsinghua.edu.cn/simple` | Tsinghua |
| `pytorch-cu128` | `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | Nanjing University, PEP503 layout |
| `nvidia` | `https://pypi.nvidia.com` | **Official** — TensorRT libs only exist here |

> [!WARNING]
> **`tensorrt-cu12-libs` must stay on the NVIDIA index regardless of your location.** The
> Tsinghua/PyPI copies are only stub sdists with no usable wheel. Do not "fix" this by moving
> it to a mirror.

### If you're outside China

Swap the mirror URLs back to the official indexes for best speed, then re-lock:

```toml
# pyproject.toml
[[tool.uv.index]]
url = "https://pypi.org/simple"          # was: tsinghua
default = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"   # was: mirrors.nju.edu.cn
explicit = true

# leave the `nvidia` index (pypi.nvidia.com) unchanged
```

```powershell
uv lock        # re-resolve against the official indexes
```

> Don't use Aliyun (`mirrors.aliyun.com/pytorch-wheels/...`) as the torch source — its flat
> layout with `+cu128` local-version tags doesn't match uv's named-index resolution and yields
> `No solution found`. NJU or SJTU (`/pytorch/whl/cu128`, a `/torch/` subdir PEP503 mirror)
> work.

### Proxy (inside China, slow links)

The first `uv sync` downloads ~2.2GB of TensorRT libraries; gvsbuild downloads many source
tarballs. If those stall:

```powershell
$env:HTTP_PROXY  = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
git config --global http.proxy  http://127.0.0.1:7890    # gvsbuild also clones git deps
git config --global https.proxy http://127.0.0.1:7890
```

> `uv sync` writes normal progress to **stderr**, so PowerShell `2>&1` may color it red even
> on success — check for "Resolved N packages" and the exit code, not the color.

---

## gvsbuild pitfalls

Compiling GTK/GStreamer via gvsbuild (`Build-SystemDependencies`) is the most fragile stage.
These are real pitfalls hit on the build machine. **All gvsbuild source edits live inside
`venv_gtk_release/Lib/site-packages/gvsbuild/` and are lost on a clean venv rebuild — you'll
need to reapply them.**

> [!NOTE]
> The background-task exit code `GVSBUILD_PS_EXIT=0` is PowerShell's code, **not** proof that
> gvsbuild succeeded (the Python traceback isn't propagated). Always verify the real artifacts
> under `build_gtk_release/gtk/x64/release/` (e.g. `bin/gdbus.exe`, `bin/msgfmt.exe`, the sink
> plugin `bin/../lib/gstreamer-1.0/gstgtk4.dll`).

### 1. Large downloads stall (network)
gvsbuild's downloader has no resume. On international links, large tarballs (cairo 32MB,
openssl 55MB, harfbuzz 18MB) stall and exhaust gvsbuild's retries.
**Fix**: set the 7890 proxy (above). For a stuck file, resume manually with
`curl -C - -L -x http://127.0.0.1:7890 -o <pkg> <url>`, verify (`xz -t` / `gzip -t`), drop into
`build_gtk_release/src/` — gvsbuild skips it on the next checksum pass. The cache accumulates;
don't restart the whole build.

### 2. gettext fails — U1052 `gettext-runtime-objs.mak` not found
Caused by the registry key `NoDefaultCurrentDirectoryInExePath=1`, which stops cmd.exe from
resolving the bare `create-lists.bat` name that gettext's nmake script relies on.
**Fix**: edit `gvsbuild/projects/gettext.py` `build()` to append the absolute nmake directory
(containing `create-lists.bat`) to `exec_vs(..., add_path=...)`. Verify `msgfmt.exe` appears.

### 3. gobject-introspection fails — setuptools 81
`Compiler.__init__() takes from 1 to 3 positional arguments but 4 were given` (setuptools 81
changed vendored distutils). The script pins `setuptools<81` only for `venv_release_win`;
`venv_gtk_release` pulls 82.
**Fix**: `venv_gtk_release\Scripts\python.exe -m pip install "setuptools<81.0.0"`, then clean
the g-i build dir and rebuild.

### 4. librsvg / gst-plugin-gtk4 — cargo-c needs a newer rustc
`cargo install cargo-c --locked` grabs 0.10.23 (needs rustc ≥1.94) but gvsbuild's toolchain is
1.92 → exit 101.
**Fix**: in **both** `gvsbuild/projects/librsvg.py` and `gvsbuild/projects/gstreamer.py`
(`GstPluginGtk4.build`), pin `install cargo-c --locked --version 0.10.21`.

### 5. libvpx — `/tmp` mismatch (Git coreutils on PATH)
configure reports `cat: /tmp/vpx-conf-*.c: No such file`. gvsbuild's `__minimum_env`
deliberately keeps Git's `usr\bin` on PATH; libvpx then calls Git Bash's `cat`/`mv` (whose
`/tmp` differs from MSYS2's).
**Fix**: in `gvsbuild/projects/libvpx.py` `build()`, temporarily strip the Git install bin dir
from `self.builder.vs_env["PATH"]` and prepend MSYS2's, restore in `finally`. Match only the
Git install bin (`\git\(usr\bin|bin|cmd|mingw\d+\bin)`) — a blanket `\git\` match also nukes the
project path `D:\Git\lada-realtime\...nasm`.

### 6. git-based deps slow to clone
`git config --global http.proxy http://127.0.0.1:7890` (+ https).

### 7. gst-plugins-base built without GL → gst-plugin-gtk4 fails
gtk4paintablesink needs `gstreamer-gl-1.0`, but gst-plugins-base's `gl_*=auto` fails to detect
a winsys and skips it.
**Fix**: in `gvsbuild/projects/gstreamer.py` `GstPluginsBase.build`, add
`-Dgl=enabled -Dgl_api=opengl -Dgl_platform=wgl -Dgl_winsys=win32`. (Triggers a downstream
rebuild.)

### 8. ffmpeg (gst-libav) configure fails — cl 19.44
cl 19.44 dropped `-o`, corrupting configure's compiler probe (`C compiler test failed` /
LNK1136), plus `-std=c11` incompatibilities with ffmpeg 8.0.1.
**Workaround used**: the dev `build_gtk/` (a *separate* GTK tree from `build_gtk_release/`) has
a working `gstlibav.dll` + ffmpeg runtime (avcodec-62/avformat-62/avutil-60/avfilter-11/
swscale-9) built with an older toolchain, and both GStreamer trees are 1.26.10 (byte-identical
`gstreamer-1.0-0.dll`). Copy `gstlibav.dll` into `lib/gstreamer-1.0/` and the 5 av*/sw* DLLs
into `bin/`; verify with `dumpbin`. Note the release sink plugins are `gstgtk4.dll` /
`gstlibav.dll` (**no `lib` prefix** — don't look for `libgstgtk4.dll`).

---

## Testing the build

After a packaging change that touches system dependencies, **test the exe on a pristine
Windows VM**. This confirms PyInstaller picked up everything available on the build machine
but possibly absent on a user's machine.

For TRT specifically, verify on a clean machine:

1. `dist/lada/_internal/tensorrt_libs/nvinfer_builder_resource_10.dll` exists (~1.79GB) —
   confirms `collect_all` worked.
2. Launching the GUI → first-run "Build acceleration engines" no longer errors with
   `No module named 'tensorrt'` and proceeds to compile.

---

## Publishing

- Attach the split `lada-<version>.7z.001` / `.002` to the GitHub Release draft.
- Upload the single-file `lada-<version>.7z` to https://pixeldrain.com.
- Add the Pixeldrain link + GitHub Release link to both the GitHub and Codeberg draft releases.
- After updating gvsbuild/uv/python: bump versions in `package_executable.ps1`, update
  `patches/`, do a `-cleanGvsbuild` rebuild, refresh download links in `windows_install.md`.
