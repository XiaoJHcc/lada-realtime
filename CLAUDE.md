# CLAUDE.md

本文件为 Claude Code 等 AI 助手提供本仓库的工作指引。代码标识符、文件路径、技术术语保留英文；说明性文字用中文。

## 项目是什么

本仓库是 [lada](https://codeberg.org/ladaapp/lada)（AI 视频去马赛克工具）的 fork。**原项目的设计目标是「处理并导出一个新视频文件」**，其内置的实时播放（GUI 的 Watch 标签页）仅作为预览，没有针对实时性做优化 —— 模型跟不上时它会**暂停并缓冲**。

**本 fork 的目标**：重写 / 新建一个**真正的实时预览**窗口，调度模型从「数据驱动（每一帧都要等到 AI 处理完、不够就停下缓冲）」改为「**时钟驱动**（播放进度永不停顿；显卡跟得上就显示 AI 去码结果，跟不上就回退到原片或降分辨率，待显卡追上）」。详见 README 的 fork 意图。**该功能尚在开发中。**

> **实时改造进度**：时钟驱动实时预览已基本成型 —— 独立路径在 `lada/gui/realtime/`(与 `lada/gui/watch/` 平级),含时钟驱动 appsrc、管线管理器、独立的 Realtime 标签页。播放墙钟驱动、永不缓冲停顿、AI 帧未就绪时回退原片。已落地的关键机制:
> - **处理前沿闸门**(`MosaicDetector` / `FrameRestorer` 的 `set_processing_frontier`,**默认关闭** —— CLI 导出与 watch 不调用即等同上游行为):realtime appsrc 每推一帧驱动闸门,让 AI 只处理「播放头前方 N 帧」,避免 detector/YOLO 全速冲到片尾抢占 GPU、把算力浪费在会被丢弃的未来帧上。realtime 用模块开关 `REALTIME_FRONTIER_GATE_ENABLED` 控制(默认 True)。
> - **clip-based AI 调度 + 冷启动超前量**(取代旧的 `realtime_preheat_duration` 秒数模型):seek 后 passthrough 从落点起播(先放原片),AI restorer 从「落点 + `realtime_cold_start_clips` × `realtime_clip_length`」起,播放头到达时去码帧就绪 → 无缝切去码。落后时后台线程重定位(`REALTIME_REPOSITION_ENABLED`,**当前默认 False** —— 曾单独引起闪帧,待帧号修复后复测)。详见 memory `realtime-clip-scheduling`。
> - **AI 超前窗口/缓冲窗口**(config `realtime_lookahead_frames`,帧,UI 名「Buffer window/缓冲窗口」,默认 180):闸门允许 AI 领先播放头的帧数 = 简单段可囤多少去码帧给难段消费。用帧数因为它与 clip 长度、内存挂钩、且 fps 不定。
> - **模型预热**(`BasicvsrppMosaicRestorer.warmup`,在 `restorationpipeline/load_models` 加载后调用):dummy clip 跑一次 forward,把 CUDA/cuDNN 初始化(kernel 懒编译、allocator 首次大块分配)在加载期付清。**2026-06 实测:warmup(T=8) 已足够,clip0 forward ≈ clip1(无「首个满长 clip 特别慢」),加大 warmup_T 对首帧无改善** —— 即此机制确实生效,但不是冷启动延迟的来源。模型在 provider 缓存 → 只付一次。
> - **帧号锚定真实 PTS**(`FrameRestorer.start` → `video_utils.first_decoded_frame_num_after_seek`):PyAV BACKWARD seek 回退到关键帧,旧代码把关键帧硬标成标称 start_frame → 帧号高估一个 GOP,污染进度条/闸门 → 整段丢弃。现按真实解码 pts 算 start_frame。CLI(`start_ns=0`)行为不变。
> - **输出队列可配**(`FrameRestorer(frame_restoration_queue_max_bytes=...)`,默认 512MB = 上游;realtime 按窗口放大并封顶 3GB 主机内存):去码帧是 CPU tensor,放大占内存不占显存。
> - **诊断卡片**(realtime 设置页「Realtime playback/实时播放」组,`ConfigSidebar.show_realtime_playback` 门控):检测/修复帧率(按批/clip 计时,reposition 不清零) / 缓冲窗口横条(`BufferBar`) / AI 命中率 / 丢弃帧数,作为调参仪表盘。
>
> `lada/gui/watch/` 下的上游 buffer-first 路径**完全不动**,作为对照保留。
> 下面「去码管线数据流」「卡顿根源」「GUI 现有应对方式」描述的是**上游 watch 路径**的架构 —— 仍然准确,且是实时路径复用/对照的基础。

### 实时改造:待解决问题(下次从这里继续)

> **2026-06 性能实测**:用 `test_video.mp4`(1080p h264 30fps)+ 离线驱动真实 `FrameRestorer`(脚本 `scripts/realtime_coldstart_profile.py`、`scripts/degrade_knob_probe.py`,模拟播放头驱动 frontier gate,5 个 seek 点取均值)做了一轮拆解。**多个「听起来合理」的方向被实测否掉,以下表格封死无效方向,避免重走**。关键背景数:4080 + fp16 + clip_size 256 + basicvsrpp-v1.2,VSR restorer **独占** GPU 时 forward ≈ **62fps**;与 YOLO detector **共享**(detector_lead=2*clip,即现状)时 restorer 实测 forward ≈ **45fps** → **YOLO 争用吃掉约 27% restorer 吞吐**。冷启动首帧延迟:clip30 ≈ 2.3s,clip15 ≈ 1.9s。

| 方向 | 状态 | 实测依据 |
|---|---|---|
| **warmup 形状对齐真实 clip 长度** | ❌ 无效 | clip0 forward ≈ clip1(634 vs 670ms),无「首个满长 clip 特别慢」的一次性成本。warmup_T=8→30 对首帧无改善。当前 warmup(T=8)已足够付清 CUDA 初始化。 |
| **detector_lead 2*clip→1*clip** | ❌ 净亏 | forward-fps 升到 60(GPU 少被抢),但 **delivered(墙钟交付)从 45 掉到 40fps** —— detector 只领先 1 个 clip,restorer 在 clip 边界空等 detector。1*clip 稳态净亏,不可用。 |
| **detector_lead 改 1.2*clip(冷启动甜点)** | ✅ 已落地 | clip_T=30 实测(n=5 配对):冷启动首帧 2.0x≈2227ms→**1.2x≈2131ms**(稳赚 ~96ms,所有 seek 一致、区间不重叠),1.0x 又涨回 ~2220(空等)。稳态 delivered 2.0x=40.7 vs 1.2x=39.0(差落在噪声内,t≈1.3);2.0x vs 1.0x 才是真差(~3.5fps)。默认已改 [frame_restorer.py:97](lada/restorationpipeline/frame_restorer.py#L97) `round(1.2*max_clip_length)`,硬下界 ≥max_clip_length,只走 realtime。 |
| **VSR clip_size 降到 256 以下** | ❌ **跑不了** | basicvsrpp-v1.2 内部 4× 下采样,要求 low-res ≥ 64 → 输入 < 256 直接 assert 失败(224→56<64)。**CLAUDE.md 旧版「256→192 省 44%」对当前模型不成立**。要降需换模型/重训,非调参。 |
| **YOLO 降 imgsz** | ⚠️ 收益小且非单调 | 640/512/384/320 → 530/463/321/338 fps。YOLO 单体本就 530fps(远超 30fps 需求),瓶颈是它**占用 GPU 时间片**而非算得慢;降 imgsz 省的卷积时间被固定的 NMS+process_mask 后处理淹没。 |
| **YOLO 挪到核显(UHD 770)释放 4080** | ❌ 高风险净亏(未实测,强推理) | UHD 770 ≈ 0.7 TFLOPS vs 4080 ≈ 49;且整段 NMS/process_mask 后处理在 detector device 上。iGPU 几乎确定 < 30fps 检测 → detector 变瓶颈 → 落入上面 detector_lead 同款空等陷阱,比现状更差。 |
| **冷启动短 clip 爬坡** | ✅ 有效、零稳态代价 | clip15 首帧 1.9s vs clip30 2.3s;且 clip15/clip30 稳态交付吞吐相当(均 ~45fps),**无 spynet 短 clip 惩罚**(旧版担心的混淆变量已排除)。 |
| **YOLO 跳帧 / 稀疏检测** | ❓ **唯一未验证的高潜力项** | 逻辑:detector forward 次数 ÷N,直接减少 4080 占用 → 把 restorer 从 45 推向 62(上限 +27%)。马赛克区域帧间移动小,复用 mask 理论几乎不掉质量。**收益上限 = 那 27%,需实测跳帧后交付吞吐 + 质量**。 |

按优先级,下次开工从这里接:

1. **难段稳态瓶颈 = YOLO 与 restorer 共享 4080 的争用(已重新定性)**:旧版写的「难段撞 30fps 硬墙、余量≈0」**基于未启用 gate 的失真测量,已撤回**。真实图景:restorer 独占可达 62fps,共享降到 45fps。瓶颈不是 VSR 物理算力不足,而是 detector 占用了约 27% 的 GPU 时间。**正路是 #2 的 YOLO 跳帧**(把这 27% 要回来),而非降 VSR 分辨率(跑不了)或挪 iGPU(净亏)。
2. **YOLO 跳帧/稀疏检测(未实现,提稳态吞吐的唯一活路)**:detector 隔 N 帧跑一次推理,中间帧复用上一次的 mask/box。需实测:跳帧 N=2/3 后 restorer 交付吞吐能否从 45 抬向 60、对 clip 边界/scene 聚合的影响、以及跳帧对检测质量的实际损失(马赛克移动小,理论可接受)。改动点在 `MosaicDetector._frame_feeder_worker` / `_frame_detector_worker` 的逐帧检测逻辑。
3. **冷启动短 clip 爬坡(已验证有效,可直接做)**:restorer 启动后前 N 个 clip 用较短的 max_clip_length(如 15),之后恢复到 `realtime_clip_length`,首帧从 ~2.3s 压到 ~1.9s,零稳态代价。改动点在 `MosaicDetector._create_clips_for_completed_scenes` 让 max_clip_length 在冷启动期递增;只走 realtime,CLI/watch 不受影响。
4. **reposition 复测(挂起)**:自动重定位 `REALTIME_REPOSITION_ENABLED` 当前默认 False —— 它曾单独引起「闪一帧别处画面」。帧号已锚定真实 PTS、模型已预热后,理论上重启的 seek 落点也对齐了,需开回 True 单独验证是否还闪/还反复失败。
5. **统一解码源(优化项)**:realtime 现在 passthrough 与 AI 源各自 `VideoReader` 解码同一文件(两遍解码)。可考虑单解码源 + 共享。

> **关于 `first_decoded_frame_num_after_seek`(成本 C)**:每次 seek 持锁同步额外 open+seek+decode 一帧只为算 start_frame,实测 ~53ms/seek。不是冷启动主因,但纯加在关键路径、可白赚(让 feeder 解出首帧后回填帧号)。优先级低于 #2/#3。

下面「去码管线数据流」等小节描述的是上游 watch 路径架构,仍准确,是实时路径复用/对照的基础。


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
    config/                侧边栏设置（device / model / max_clip_duration / preview_buffer_duration / realtime_clip_length / realtime_cold_start_clips / realtime_lookahead_frames 等）
    watch/                 ★ 上游 buffer-first 预览窗口（保留作对照，不动）
      gstreamer_pipeline_appsrc.py    ★ FrameRestorerAppSrc：把 AI 帧推进 GStreamer
      gstreamer_pipeline_manager.py   ★ 整条 GStreamer 管线 + 缓冲队列策略
      watch_view.py                   ★ 播放/暂停/seek UI 逻辑 + 缓冲自适应
      timeline.py, seek_preview_popover.py, overlay_elements_controller.py
    realtime/              ★★ 时钟驱动实时预览（本 fork 主战场，与 watch/ 平级）
      gstreamer_pipeline_appsrc_realtime.py  ★★ RealtimeFrameRestorerAppSrc：时钟驱动推帧 + passthrough 回退 + 处理前沿/clip-based 调度/冷启动超前量/超前窗口 + 诊断埋点
      gstreamer_pipeline_manager_realtime.py ★★ RealtimePipelineManager：无 min-threshold 缓冲，暴露诊断/clip/冷启动/窗口 setter
      realtime_view.py                ★★ Realtime 标签页 UI（复用 watch 的 overlay/seek-preview，精简缓冲逻辑）
    export/                导出页面 UI
  restorationpipeline/     ★★★ 去码核心管线（CLI 导出与两个预览路径共用）
    frame_restorer.py            ★★★ FrameRestorer：编排 5 个工作线程 + 队列；set_processing_frontier（默认关闭闸门）
    mosaic_detector.py           ★★★ MosaicDetector：检测 + 把帧聚成 Clip（含 max_clip_length + feeder 处理前沿闸门）
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
3. **`_frame_restoration_worker` 阻塞等待**（`frame_restorer.py` 的 `_clip_buffer_contains_all_cips_needed_for_current_restoration` while 循环，约 line 383）：若当前帧有马赛克，必须等到「覆盖该帧的 clip」检测完 *并* 修复完才能输出。

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
