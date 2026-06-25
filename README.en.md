<h1 align="center">
  <img src="assets/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada Realtime
</h1>

<p align="center">
  <em>A fork of <a href="https://codeberg.org/ladaapp/lada">lada</a> focused on true real-time mosaic removal during playback.</em>
</p>

<p align="center">
  <strong>English</strong> ·
  <a href="README.md">中文</a>
</p>

> [!WARNING]
> **Work in progress.**

## Motivation

[lada](https://github.com/ladaapp/lada) is an AI mosaic-removal tool. The original is built for **offline export**, and its built-in preview is **not optimized for real-time playback** — as the upstream README puts it: To watch the restored video in real-time, you'll need a **powerful machine**.

But isn't an RTX 4080 powerful enough? Export already runs at 30–45 fps, so the potential for real-time playback is clearly there — yet the upstream's buffer-first pacing makes it impractical for actually *watching* a video.

This fork doesn't touch the models themselves. It only reworks AI task scheduling and the front-end playback strategy: keep playback going first, remove mosaics when the GPU can keep up.

## What's Done

**Playback: data-driven → clock-driven**

- **Never pause**: Upstream's pacing is dictated by AI progress — every frame waits for its restoration to finish, and it pauses to buffer when it can't keep up. This fork inverts that: playback never stops; if the AI frame is ready it shows the restored image, otherwise it falls back to the original.

> So you still need a powerful GPU — otherwise it just keeps falling back to the original.

**Scheduling: compute only what's about to be seen, and prefetch the future**

- **Prefetch**: After a seek, the AI skips the frames right at the playhead and starts a short distance ahead. The original plays for that brief gap, and by the time the playhead catches up the restored frames are ready for a seamless switch.
- **Bounded frontier**: Fixes the upstream behavior where the detector (YOLO) races ahead indefinitely; now it only processes the short window the restorer (BasicVSR++) needs for its next clip.
- **Warmup**: Run a dummy pass right after loading so the GPU's one-time initialization cost is paid during load, not on the first real frame.
- **Smaller window**: Default to a shorter clip window for faster response, at the cost of some temporal stability.

**TensorRT acceleration (Nvidia)**

- Splits the BasicVSR++ restorer into 6 TensorRT sub-engines — a measured **3–4× speedup** of restorer inference when it has the GPU to itself. Nvidia + FP16 only; everything else falls back to PyTorch seamlessly.
- Engines are bound to the GPU architecture + TensorRT version, so they are **compiled once locally on first run** (~10–15 minutes, one time). Swapping GPUs or upgrading TensorRT invalidates them and triggers a rebuild. See "Build & Distribution" below.

## Build & Distribution

From source checkout to a distributable build (Windows, Nvidia). For other platforms see upstream [`docs/`](docs/).

### 1. Checkout & system dependencies

```powershell
git clone <this-repo-url> lada-realtime
cd lada-realtime
```

The packaging script installs system dependencies automatically (FFmpeg / uv / 7zip / MSYS2 / VS Build Tools, etc.) — no manual setup needed.

### 2. One-shot packaging

[`packaging/windows/package_executable.ps1`](packaging/windows/package_executable.ps1) is end-to-end: installs system deps, builds GTK via gvsbuild, compiles translations, downloads model weights, creates a venv, installs Python deps and applies patches, runs PyInstaller, and finally produces a 7z archive:

```powershell
# Run from the project root (defaults to Nvidia)
.\packaging\windows\package_executable.ps1 -extra nvidia
```

Output is `lada.exe` (GUI) + `lada-cli.exe` (CLI), bundled into a 7z archive. Useful flags:

- `-cliOnly`: CLI only, skips the GTK build.
- `-skipWinget` / `-skipGvsbuild`: skip when system deps / GTK are already built — saves time.
- `-extra intel`: Intel Arc.

> **TensorRT acceleration dependency**: this fork's TRT acceleration needs `torch-tensorrt`. It isn't a default dependency — before packaging, run `uv pip install torch-tensorrt` inside the venv (matching your local torch, e.g. `torch-tensorrt==2.8.0`). Packaging works without it; you just get the PyTorch path at runtime.

### 3. TensorRT engines are NOT shipped

TRT engines are bound to a specific GPU architecture + TensorRT version and **cannot be distributed across machines**, so the archive contains **no** prebuilt engines — each end user compiles them on their own machine:

- **GUI**: a first-run dialog guides the user — no Nvidia GPU → notice that the PyTorch path is used; single GPU → "Build now / Later"; multiple GPUs → pick which card (the chosen card is also the inference device). After "Build now" the dialog shows compile progress in-place (~10–15 min, not cancellable). Skipping re-prompts on the next launch until built once.
- **CLI / install scripts**: run `lada-cli --build-trt-engines` once after install to prewarm, moving the compile to install time so the first playback/export doesn't block.

Engines are cached under `model_weights/<model>_sub_engines/`, with the GPU architecture, TensorRT version, and precision encoded in the filename, so upgrading `torch-tensorrt` or swapping GPUs auto-invalidates and rebuilds them. Set `LADA_BASICVSRPP_TRT=0` to force the PyTorch path.

### Running from source (development)

To run from source without packaging, see "Build & Run" in [`CLAUDE.md`](CLAUDE.md) and [`docs/windows_install.md`](docs/windows_install.md). In short: `uv venv` → `uv sync --extra nvidia` → apply patches → download models into `model_weights/` → for the GUI also have `build_gtk/` in place. A source run needs `translations/*.po` compiled to `.mo` to show Chinese (see CLAUDE.md).

## License

Source and models follow upstream, licensed under **AGPL-3.0**. Full terms in [LICENSE.md](LICENSE.md). This fork's changes are released under AGPL-3.0 as well.

## Acknowledgement

This project is a fork of **[lada](https://codeberg.org/ladaapp/lada)** by the Lada Authors — all credit for the original application, models, and training pipeline goes to them.

Lada itself builds upon the work of these projects:

* [DeepMosaics](https://github.com/HypoX64/DeepMosaics): Provided code for mosaic dataset creation. Also inspired the original project.
* [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++) / [MMagic](https://github.com/open-mmlab/mmagic): Base model for mosaic removal.
* [YOLO/Ultralytics](https://github.com/ultralytics/ultralytics): Training mosaic and NSFW detection models.
* [DOVER](https://github.com/VQAssessment/DOVER): Video quality assessment during dataset creation.
* [DNN Watermark / PITA Dataset](https://github.com/tgenlis83/dnn-watermark): Watermark detection dataset.
* [NudeNet](https://github.com/notAI-tech/NudeNet/): Additional NSFW classifier.
* [Twitter Emoji](https://github.com/twitter/twemoji): Eggplant emoji used as base for the app icon.
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Image degradation model design.
* [BPJDet](https://github.com/hnuzhy/BPJDet): Human body/head detection for creating SFW mosaics.
* [CenterFace](https://github.com/Star-Clouds/CenterFace): Face detection for creating SFW mosaics.
* PyTorch, FFmpeg, GStreamer, GTK and [all other folks building our ecosystem](https://xkcd.com/2347/)
