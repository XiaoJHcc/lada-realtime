# CLAUDE.md

本文件为 Claude Code 等 AI 助手提供本仓库的工作指引。代码标识符、文件路径、技术术语保留英文；说明性文字用中文。

## 项目是什么

本仓库是 [lada](https://codeberg.org/ladaapp/lada)（AI 视频去马赛克工具）的 fork。**原项目的设计目标是「处理并导出一个新视频文件」**，其内置的实时播放（GUI 的 Watch 标签页）仅作为预览，没有针对实时性做优化 —— 模型跟不上时它会**暂停并缓冲**。

**本 fork 的目标**：重写 / 新建一个**真正的实时预览**窗口，调度模型从「数据驱动（每一帧都要等到 AI 处理完、不够就停下缓冲）」改为「**时钟驱动**（播放进度永不停顿；显卡跟得上就显示 AI 去码结果，跟不上就回退到原片或降分辨率，待显卡追上）」。详见 README 的 fork 意图。**该功能尚在开发中。**

> 当前这套实时改造代码尚未落地。本文件描述的是**现有上游架构**，以及实时改造需要关注/改动的位置。

## 技术栈

- **Python** ≥ 3.12（Windows 打包用 3.13）。依赖管理用 [`uv`](https://docs.astral.sh/uv/)，见 `pyproject.toml` / `uv.lock`。
- **PyTorch**：推理后端。设备可为 `cuda`（Nvidia）、`xpu`（Intel Arc）、`mps`（Apple Silicon）、`cpu`。Nvidia 走 `torch==2.8.0` + cu128。
- **模型**：
  - **马赛克检测** = YOLO11 分割模型（Ultralytics，`.pt`）。
  - **马赛克修复** = **BasicVSR++**（`basicvsrpp-v1.2`，`.pth`），一个**时间维度（temporal）模型** —— 它一次吃进一段连续帧（clip）来降低帧间闪烁。也可选 DeepMosaics。
- **GUI**：GTK4 + libadwaita（PyGObject）。视频播放完全基于 **GStreamer** 管线，AI 帧通过自定义 `AppSrc` 注入，最终渲染到 `gtk4paintablesink`。
- **视频 I/O**：PyAV（解码，`VideoReader`）、FFmpeg/`av`（导出编码，`VideoWriter`）、`ffprobe`（元数据）。
- **打包**：Windows/macOS 用 PyInstaller（`packaging/windows`、`packaging/macOS`）；Linux 用 Flatpak（`packaging/flatpak`）与 Docker（`packaging/docker`）。

## 仓库结构

```
lada/
  cli/                     CLI 入口（导出用途）。main.py: process_video_file 是导出主循环
  gui/                     GTK4 GUI
    main.py, application.py, window.py
    frame_restorer_provider.py   ★ 模型加载/缓存；构造 FrameRestorer；含 PassthroughFrameRestorer
    config/                侧边栏设置（device / model / max_clip_duration / preview_buffer_duration 等）
    watch/                 ★★ 实时预览窗口（本 fork 主战场）
      gstreamer_pipeline_appsrc.py    ★★ FrameRestorerAppSrc：把 AI 帧推进 GStreamer
      gstreamer_pipeline_manager.py   ★★ 整条 GStreamer 管线 + 缓冲队列策略
      watch_view.py                   ★★ 播放/暂停/seek UI 逻辑 + 缓冲自适应
      timeline.py, seek_preview_popover.py, overlay_elements_controller.py
    export/                导出页面 UI
  restorationpipeline/     ★★★ 去码核心管线（实时卡顿的根源就在这）
    frame_restorer.py            ★★★ FrameRestorer：编排 5 个工作线程 + 队列
    mosaic_detector.py           ★★★ MosaicDetector：检测 + 把帧聚成 Clip（含 max_clip_length 逻辑）
    basicvsrpp_mosaic_restorer.py  BasicVSR++ 推理封装（restore(clip)）
    deepmosaics_mosaic_restorer.py
  models/                  模型定义（basicvsrpp/、yolo/、deepmosaics/ 等）
  utils/                   video_utils（VideoReader/Writer/seek）、threading_utils（PipelineQueue/线程/marker）、image_utils、mask_utils 等
  datasetcreation/         数据集制作（与实时无关）
configs/                   训练/模型配置
scripts/                   训练与评估脚本（与实时无关）
packaging/                 各平台打包脚本
patches/                   对第三方库（mmengine/ultralytics 等）的补丁，安装时打入 .venv
docs/                      linux/macOS/windows 安装指南、训练文档
model_weights/             模型权重（仓库内只有 *.license 占位，真权重需自行下载）
```

★ 数量代表与实时改造的相关度。

## ★★★ 去码管线数据流（必读）

`FrameRestorer.start()` 启动后，下面 5 个线程通过 `PipelineQueue`（带统计的有界队列）串成生产者/消费者链：

```
                MosaicDetector（3 线程）                        FrameRestorer（2 线程）
 video ─► [feeder] ─► frame_feeder_q ─► [inference] ─► inference_q ─► [detector] ─┬─► frame_detection_q ─┐
 (解码+YOLO预处理)        (YOLO 分割)               (聚成 Scene/Clip)          │   (每帧的检测数)        │
                                                                                └─► mosaic_clip_q ─► [clip_restore] ─► restored_clip_q ─┐
                                                                                       (BasicVSR++ 整段修复)                            │
 video ───────────────────────────────────────────────────────────────────────────────────────────────► [frame_restore] ◄──────────┘
 (再次解码原片)                                                              (把修复后的马赛克区域 blend 回原帧) ─► frame_restoration_q ─► 消费者
```

消费者：
- **CLI 导出**：`for elem in frame_restorer:`（`__next__` 从 `frame_restoration_q` 取帧）→ `VideoWriter.write()`。
- **GUI 预览**：`FrameRestorerAppSrc._appsource_worker` 从 `frame_restoration_q` 取帧 → `push-buffer` 进 GStreamer。

注意：**视频被解码了两遍** —— `MosaicDetector._frame_feeder_worker` 和 `FrameRestorer._frame_restoration_worker` 各自 `open` 文件并 `seek`（`video_utils.VideoReader`，PyAV）。

### ★ 卡顿/延迟的根源

1. **Clip 必须凑够才能修复**（`mosaic_detector.py:243` `_create_clips_for_completed_scenes`）：一段连续马赛克区域（`Scene`）只有在「**场景结束** / **达到 `max_clip_length`（默认 180 帧 ≈ 6s@30fps）** / **EOF**」时，才会被打包成 `Clip` 送去修复。
2. **BasicVSR++ 一次吃整段**（`basicvsrpp_mosaic_restorer.py: restore`）：clip 越长，时间维度越稳（闪烁越少），但首帧延迟越大。
3. **`_frame_restoration_worker` 阻塞等待**（`frame_restorer.py:329`）：若当前帧有马赛克，必须等到「覆盖该帧的 clip」检测完 *并* 修复完才能输出。

→ **seek 之后**：检测器从零开始累积最多 `max_clip_length` 帧，跑完整段 BasicVSR++，第一帧才出来。这就是「跳片段后总卡顿」。

### GUI 现有应对方式（与实时目标相反）

`gstreamer_pipeline_manager.py` 在 `AppSrc` 与 sink 之间插了个 GStreamer `queue`，设了 **`min-threshold-time`**（缓冲到这么多秒才开始播）。

- 默认 `preview_buffer_duration = 0` 表示「自动」，自动值 = `max_clip_length / fps`，并被 `watch_view.py` 钳制在 2–8s（`_buffer_queue_min_thresh_time_auto_min/max`）。
- **underrun**（缓冲见底）→ 管线 **pause**、emit `waiting-for-data=True`、自动缓冲 ×1.5（`watch_view.py:582` `on_waiting_for_data`）。
- 这就是 README 里「player may pause and buffer」的来源 —— **buffer-first**。

### 实时改造要面对的核心矛盾（设计备忘）

- **时间维度模型天然有延迟**：BasicVSR++ 必须看到一段帧。真正「零延迟逐帧实时」与该模型不兼容。实时方案需要**小 clip 窗口 / 滑动窗口**（牺牲部分时间稳定性换低延迟），延迟下界 ≈ clip 帧数。4080 导出能到 90fps 说明**吞吐不是瓶颈，延迟（clip 窗口）才是**。
- **时钟驱动调度**：以播放墙钟为准，为「当前播放时间」挑选「当前可得的最佳帧」。AI 帧没准备好 → 退回原片（已有 `PassthroughFrameRestorer` 可复用），追上后再切回。
- **降级手段**：减小 `max_clip_length`、降推理分辨率（`clip_size`，检测器默认 256）、`fp16`、跳帧推理（处理稀疏帧 + 复用）等，目标是让 GPU 持续满足 fps。
- **统一解码路径**：实时下应避免两个 `VideoReader` 各自 seek，考虑单解码源 + 共享。
- **可复用的现成件**：`PassthroughFrameRestorer`（原片直通）、`FrameRestorerProvider`（模型缓存，切模型/设备/fp16 才 reload）、`PipelineQueue` 的统计（`_dump_queue_stats` 可定位瓶颈队列）。

## 构建与运行（从源码跑起来）

环境为 Windows。详细步骤见 `docs/windows_install.md`，要点：

```powershell
# 1. 系统依赖（管理员 PowerShell）
winget install --id Gyan.FFmpeg -e --source winget
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget

# 2. 虚拟环境 + Python 依赖（按显卡选 extra：nvidia / nvidia-legacy / intel / cpu）
uv venv
.\.venv\Scripts\Activate.ps1
uv sync --extra nvidia          # RTX 4080 用 nvidia
uv run --no-project python -c "import torch; print(torch.cuda.is_available())"   # 期望 True

# 3. 打补丁（mmengine / ultralytics 等，打入 .venv）
uv pip install patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
uv pip uninstall patch

# 4. 下载模型权重到 model_weights/（仓库内只有 .license 占位，真权重需下载）
#    至少需要：检测模型 v4-fast / restoration 模型 generic v1.2，见 docs/windows_install.md
```

CLI 直接可用：`lada-cli --input <video>`。

**GUI 额外需要 GTK/GStreamer 系统依赖**（`build_gtk/`，预编译包或 gvsbuild 自行编译），再装 PyGObject/pycairo wheel，然后 `lada`。`lada/gui/__init__.py` 的 `prepare_windows_gui_environment()` 在 import 时自动把 `build_gtk/gtk/x64/release/bin` 接入 `PATH`，无需手动设 typelib/插件路径。实时预览改造主要在 GUI（Watch 页），跑 GUI 是验证前提。

**Windows 打包成 .exe**：`packaging/windows/package_executable.ps1`（PyInstaller，spec 在 `packaging/windows/lada.spec`）。产物为 `lada.exe`（GUI）和 `lada-cli.exe`。

入口点（`pyproject.toml [project.scripts]`）：`lada` → `lada.gui.main:main`，`lada-cli` → `lada.cli.main:main`。版本：`lada/__init__.py` `VERSION`（当前 `0.11.1-dev`）。

### 日常最简运行流程（环境已装好后）

前提：`.venv` 已建好并 `uv sync`、patches 已打、`model_weights/` 已下载（至少 `v4_fast` + `generic_v1.2`）、GUI 还需 `build_gtk/gtk` 就位且 pygobject/pycairo wheel 已装进 `.venv`。

```powershell
.\.venv\Scripts\Activate.ps1

# CLI（导出）
lada-cli --input <video>

# GUI（实时预览改造的验证入口）—— 调试时加 LOG_LEVEL
$env:LOG_LEVEL = "DEBUG"
lada                                  # 或 python -m lada.gui.main
```

- **GUI 显中文**：默认跟随系统语言（Windows 读 `GetUserDefaultUILanguage()`），但源码版需先把翻译 `.po` 编译成 `.mo` 才有中文，否则 fallback 英文。一次性编译（用 `build_gtk` 自带 msgfmt，免动执行策略）：
  ```bash
  MSGFMT="build_gtk/gtk/x64/release/bin/msgfmt.exe"
  for lang in zh_CN zh_TW; do
    mkdir -p "lada/locale/$lang/LC_MESSAGES"
    "$MSGFMT" "translations/$lang.po" -o "lada/locale/$lang/LC_MESSAGES/lada.mo"
  done
  ```
  `LANGUAGE` / `LANG` 环境变量可覆盖系统语言。`zh_CN.po` 覆盖不全，界面会夹少量英文。
- **别和编译版 `lada.exe` 同开**：两者共用 GApplication app-id，单实例机制会让后启动的进程把窗口激活信号转发给已运行实例后**静默退出**——会误以为在跑源码版、实际在操作编译版。排查「启动即退/跑错版本」先查 `tasklist /fi "imagename eq lada.exe"` 和正在跑的 `python.exe` 命令行。

## 约定与注意事项

- 帧在管线内是 **`torch.Tensor`，格式 BGR，shape (H, W, C)，uint8**，可能在 GPU 上。`AppSrc` 推给 GStreamer 前会拷回 CPU 并按 RU4 做宽度 padding（`GstPaddingHelpers`）。
- 线程通过 `threading_utils` 的 `EOF_MARKER` / `STOP_MARKER` 哨兵在队列里传递终止信号；停止逻辑（`FrameRestorer.stop` / `MosaicDetector.stop`）靠「塞 stop marker 解阻塞消费者 + 清空队列解阻塞生产者 + join」，改动管线时务必保持这套握手，否则 join 会卡死。
- `LOG_LEVEL` 环境变量控制日志（默认 `WARNING`，调试设 `DEBUG`）。
- 许可证 **AGPL-3.0**；新增源文件沿用现有 SPDX 头：
  `# SPDX-FileCopyrightText: Lada Authors` / `# SPDX-License-Identifier: AGPL-3.0`。
- 上游主仓在 **Codeberg**，GitHub 为镜像。本仓库是个人 fork。
- 改 GStreamer 管线/缓冲行为时，注意 Windows + Nvidia + OpenGL paintable 有已知的颜色错乱 workaround（`gstreamer_pipeline_manager.py:268`，win32 上不走 glsinkbin）。
```
