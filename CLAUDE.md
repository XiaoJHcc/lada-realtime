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
> - **TensorRT 加速 BasicVSR++**(`feat/trt-basicvsrpp` 分支,2026-06 移植 jasna 做法):把 BasicVSR++ 拆成 6 个 TRT 子引擎(4 loop_body 静态 batch=1 + preprocess + upsample 动态 batch),restorer **独占** GPU 时 forward 实测 **3–4x**(T=16 时 3.16x、T=60 时 4.47x,数值对齐 MAE≈0.001)。集成在 `BasicvsrppMosaicRestorer(split_forward=...)`,`load_models` 经 `_maybe_build_trt_split_forward` 装配;模块开关 env `LADA_BASICVSRPP_TRT`(默认 on)、引擎上界 `LADA_BASICVSRPP_TRT_MAX_CLIP`(默认 180)。非 cuda / 非 fp16 / 引擎缺失 / 编译失败**无缝回退 PyTorch**,对 CLI 与 watch 透明。引擎绑 **GPU 架构 + TRT 版本 + 精度 + OS + clip 上界**(全编进文件名,如 `loop_body_backward_1.sm89.trt1012.fp16.win.engine`),**不入库不跨机分发**;首跑本地编译、缓存进 `model_weights/<stem>_sub_engines/`。注意 3–4x 是 restorer 独占数,真实管线 detector(YOLO)争用会打折 —— 见下「待解决问题」。详见 `docs/trt_basicvsrpp_port_design.md` 与 memory `jasna-trt-acceleration-reference`。
> - **TRT 可分发改造**(2026-06-25,见 memory `trt-distribution-design`):①**缓存键自愈** —— 文件名编全 arch+TRT 版本等 6 维,升级 torch-tensorrt / 换 GPU / 跨机拷 `model_weights/` 后旧引擎自动判失活并重编,不再反复加载失败;`load_sub_engines` 反序列化失败会删当前键引擎并回退 PyTorch(下次重编)。②**首跑进度文字** —— `lada/restorationpipeline/progress.py` 进程级回调通道,`load_models` 4 阶段 + 编译 6 条子消息流到 realtime spinner 页的 `label_loading_status`。③**预热命令** —— 装好后跑一次 `lada-cli --build-trt-engines`(固定用 `LADA_BASICVSRPP_TRT_MAX_CLIP`/180,与运行时一致),把几分钟编译从「首次播放/导出时阻塞」挪到安装阶段;非 cuda/非 fp16/非 basicvsrpp 优雅退出。④**首次启动引导弹窗** —— `lada/gui/trt_setup_dialog.py`(`Adw.Dialog` + Stack 两页,`window.py on_realize` 触发):无卡/fp16 关→通知走 PyTorch;单卡→build/later;多卡→下拉选卡(选定卡 = 编译 + 推理 device)。点 build 弹窗内显示编译进度(不可取消),跳过则下次启动再弹(弹与否由引擎齐全性驱动,编过一次即不再弹)。与首次开视频的自动编译兜底共存。**未做**:`hardware_compatible=True`(通用引擎、需先实测掉速)、预编下载矩阵。
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
  - **马赛克修复** = **BasicVSR++**（`basicvsrpp-v1.2`，`.pth`），一个**时间维度（temporal）模型** —— 它一次吃进一段连续帧（clip）来降低帧间闪烁。（原项目可选的 DeepMosaics 修复模型已在本 fork 移除。）
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
  models/                  模型定义（basicvsrpp/、yolo/ 等）
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

### 分发包体积分析与裁剪（2026-06 调研）

完整 nvidia 包约 **8G**（本机实测 dist 7.6G，本轮裁剪后 **6.9G**）。已逐项验证各大块的「可裁性」，按删除风险分三类。**🔴 类基于 PE 导入表硬验证（`pefile` 解析 `torch_cuda.dll` 等的静态导入表，脚本思路见下）+ cuDNN 子库实删硬崩，封死无需重查；🟡 类需逐个删除后跑真实 TRT 去码验证不崩才能坐实。2026-06-27 一轮把 `cudnn_adv`+`cusolverMg`+`curand`（DLL，共 498M）与 `polars`+`matplotlib`（模块，141M）实测安全并落地（见下表、`lada.spec`、`package_executable.ps1`）；cuDNN 其余引擎库实测硬崩归 🔴，scipy 实测需改代码归搁置 —— 🟡 已基本清空，裁剪到顶。**

> **两个常见误解先纠正**：①「PyTorch 回退可裁、用户要回退去用原版」——**不成立**。TRT 实时路径 + YOLO 检测器都跑在 torch 上，`torch.Tensor` 是管线数据格式，torch 是地基不是可选模块。②「CLI 可裁省空间」——CLI 与 GUI **共用整个 `_internal/`**，无独占大依赖，删 `lada-cli.exe` 只省 ~46M。

**🟢 确定可裁（代码零引用或函数内惰性 import，推理路径不碰）：**

| 项 | 大小 | 状态 | 依据 |
|---|---|---|---|
| `model_weights/*_sub_engines` | ~496M | ✅ 已处理 | 本机 GPU 架构的 TRT 引擎缓存，构建后在 dist 跑过一次就漏进包；对用户无效（他们首次启动自编）。`package_executable.ps1` 的 `Create-7ZArchive` 压缩前清理。 |
| polars | 129M | ✅ 已删（exclude，已出包验证） | ultralytics 训练/benchmark/plotting 路径函数内 `import polars`，lada 推理零引用。`lada.spec` `COMMON_EXCLUDES`。**踩过的坑**：旧 dist 里 `polars/polars.pyd` 仍 129M 一度让人以为 excludes 没生效 —— 实为那个 dist 构建于 COMMON_EXCLUDES 加入之前（HEAD 的 spec 没有该机制）。**2026-06-27 重新出包确认 `excludes=["polars"]` 干净生效**（新构建 `_internal/polars` 不存在）。`package_executable.ps1` 仍保留构建后删 `_internal/polars` 作廉价兜底（新构建里它是 no-op）。 |
| matplotlib | 12M | ✅ 已删（exclude + 构建后清理） | 同 polars，ultralytics 函数内惰性 import，lada 零引用。`lada.spec` `COMMON_EXCLUDES` 排除 + `package_executable.ps1` 构建后删兜底。实测坐实：import 阻断器让 matplotlib 一律 `ModuleNotFoundError` 跑完整 `load_models`+restore+detect 链不崩（`scripts/exclude_sim_verify.py`）；**无启动钩子可安全后删**，当前 dist 已手删、冻结 exe 完整导出不崩。 |
| tcl/tk（`_tcl_data`+tcl86t+tk86t） | ~8M | ✅ 已排除（仅构建期生效） | `excludes=["tkinter"]` 后 PyInstaller 既不收集数据/DLL，也不注入 `pyi_rth__tkinter` 运行时钩子。**关键坑：不能手删已构建 dist 的 `_tcl_data` —— 旧构建仍带该钩子，启动即 `FileNotFoundError: Tcl data directory ...\_tcl_data not found`。tcl/tk 只能靠 spec 在构建期排除，不能事后手删。** |
| `torch/lib/cusolverMg64_11.dll` | 149M | ✅ 已删（实测安全） | 多 GPU dense LAPACK 求解器，推理路径不碰。两路验证：① PE 导入表扫 dist 全 37 个 torch DLL 无一静态导入；② 从 `torch/lib` 移走后跑真实 restore+detect（TRT 路径 + PyTorch 回退）均有限输出不崩（`scripts/dll_trim_verify.py`），且**真实冻结 `lada-cli.exe` 完整导出整段 test_video（3576 帧 100%）不崩**。本机无 CUDA toolkit/无系统副本，排除 PATH 回退掩盖。`lada.spec` `STRIP_TORCH_DLLS` 构建期 strip；当前 dist 已手删。 |
| `torch/lib/curand64_10.dll` | 68M | ✅ 已删（实测安全） | CUDA RNG，CNN 推理不触发。验证同上（与 cusolverMg 一起测）。 |
| `torch/lib/cudnn_adv64_9.dll` | 269M | ✅ 已删（实测安全） | cuDNN「advanced」引擎库（RNN / multi-head-attention / CTC / fused ops）。去码全是纯 CNN（YOLO 卷积 + BasicVSR++ 卷积/deform conv/SPyNet 光流），无 RNN/attention → cuDNN 不会动态加载它。验证：移走后 restore（PyTorch 路径 T=8/16/64）+ YOLO detect + **冻结 exe 完整导出整段视频**全过；本机无系统副本兜底。`lada.spec` `STRIP_TORCH_DLLS` strip；当前 dist 已手删。**唯一可删的 cuDNN 子库 —— 其余引擎库删了硬崩（见 🔴）。** |

**🟡 仅剩需改代码才能动的：**

| 项 | 大小 | 风险 | 备注 |
|---|---|---|---|
| scipy | 52M(+libs 20M) | ❌ 不可直接排除（需改代码，已搁置） | **实测排不掉**：`load_model`→`register_all_modules()`→`mosaic_video_dataset`(训练模块，顶层 import)→`transforms`→`degradations.py` `from scipy import special`，BasicVSR++ **模型加载期**即把训练侧 degradations 拉进来。`exclude_sim_verify.py scipy` 在 `load_models` 阶段直接 `ModuleNotFoundError`。要省须把训练专用 import 从推理注册路径延迟化（动 `register_all_modules` / `mosaic_video_dataset` / `transforms`），属改加载/注册路径，风险高于收益，暂搁。 |

**🔴 封死（加载时刚性依赖 / 方案地基，无裁剪空间）：**

- **torch 核心 CUDA DLL ~1.7G**：`cublas64_12`/`cublasLt64_12`(644M)/`cufft64_11`(264M)/`cusolver64_11`(216M)/`cusparse64_12`(362M)/`cudnn64_9` 全在 `torch_cuda.dll` 的 **PE 导入表**里 → Windows loader 启动即解析，删任一个 `import torch` 直接 `DLL load failed`。`nvjitlink`(74M)被 `cusparse` 静态依赖，`cudart`/`cupti` 被 `torch_cpu` 依赖。
- **cuDNN 引擎子库 ~690M（precompiled 490M / ops 120M / heuristic 54M / runtime_compiled 19M / cnn 4.6M / graph 2.4M）**：cuDNN 卷积执行时按 plan 动态 `LoadLibrary` 这些子库，**严格加载、无优雅回退**。实测删 `cudnn_engines_precompiled64_9.dll`(490M) 或 `cudnn_heuristic64_9.dll`(54M) 任一个，YOLO/BasicVSR++ 一跑就 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` 硬崩——**连默认 TRT 路径也崩**（YOLO 检测器用 cuDNN 卷积，与 restorer 走不走 TRT 无关）。卷积是去码地基，全部保留。唯一例外 `cudnn_adv64_9.dll`（RNN/attention，269M）已删，见 🟢。
- **tensorrt_libs 2.2G**（其中 `nvinfer_builder_resource` 单文件 1.7G）：保留 —— 用户机现编引擎（`--build-trt-engines` / 引导弹窗）必需，引擎绑 GPU 架构无法跨机预编，TRT 方案固有成本。
- **ffmpeg.exe + ffprobe.exe 各 142M**：两者都在用（[video_utils.py:142](lada/utils/video_utils.py#L142) 读元数据、[audio_utils.py:41](lada/utils/audio_utils.py#L41) 读音频编码），且是两个独立 Gyan 静态构建（sha256 不同），无法去重。

**验证手段（复用）**：
- **PE 导入表硬验证**（`venv_release_win` 自带 `pefile`）：判断某 DLL 是否「加载时刚性依赖」= 解析依赖它的 torch DLL 的 `IMAGE_DIRECTORY_ENTRY_IMPORT`，目标 DLL 在导入表里即刚性、删了崩；不在表里则可能是运行时 dlopen（如 cuDNN 子库），需实跑验证。一次性脚本思路见本轮 `cusolverMg`/`curand` 验证。
- **运行时去码验证脚本**（本轮新增，验剩余 🟡 直接复用）：`scripts/dll_trim_verify.py` —— `load_models`（内部已 warmup BasicVSR++ + YOLO）+ 真实 clip restore + 真实帧 detect，检查有限输出；配合 shell 把目标 DLL 从 `.venv/Lib/site-packages/torch/lib` 改名移走→跑→还原（trap 保证还原）。`scripts/exclude_sim_verify.py <模块名>` —— 用 meta-path 阻断器模拟 PyInstaller `excludes`，验「排除某 Python 模块是否破坏加载链」（无需动文件）。
- **冻结产物烟测**（最终确认）：从项目根跑 `dist/lada/lada-cli.exe --input test_video.mp4 --output ... --device cuda`，frozen 模式权重目录是 bundle 内 `_internal/model_weights`（运行时钩子指定），把 `model_weights/*_sub_engines` 拷进去可跳过现编。**注意系统若装了 CUDA toolkit，PATH 上的同名 DLL 会掩盖真实依赖 → 验证机须无 CUDA toolkit / 无系统副本（本机已确认）。**

小结：本轮已落地（2026-06-27）——
- **当前已构建 dist（手删，立即生效，7.6G→6.9G，约 640MB）**：`cudnn_adv64_9.dll`(269M) + `cusolverMg`(157M) + `curand`(72M) DLL + `polars`(129M) + `matplotlib`(12M)，均冻结 `lada-cli.exe` 完整导出整段 test_video（3576 帧）验证不崩。tcl/tk(8M) 不能手删（tkinter 启动钩子），留待下次构建。
- **构建期固化（下次构建自动生效，三套机制按可靠性分工）**：① DLL（cusolverMg/curand/cudnn_adv）走 `lada.spec` `STRIP_TORCH_DLLS`（直接过滤 binaries TOC，**确定**）；② tkinter 只能走 `lada.spec` `COMMON_EXCLUDES`（排除即不注入 `pyi_rth__tkinter` 钩子，**唯一可行路**）；③ polars/matplotlib 走 `COMMON_EXCLUDES` **＋** `package_executable.ps1` 构建后删 `_internal/{polars,matplotlib}`（**双保险** —— 因 `excludes` 实测可能被 hook/改名分发打败，polars 就没排掉）。下次构建总省约 **648M**（DLL 498M + polars 129M + matplotlib 12M + tcltk 8M），另叠加已有的 sub_engines 清理。
- **已出包验证（2026-06-27，复刻 `Create-EXE` 完整 GUI 构建）**：① `STRIP_TORCH_DLLS` 生效 —— 日志两次 `-> [trim] dropped 3 torch DLL(s) (cudnn_adv64_9.dll, curand64_10.dll, cusolvermg64_11.dll)`（cli_a + gui_a 各一），新 dist 三个 DLL 全不存在；② `COMMON_EXCLUDES` 干净生效 —— 新 dist 里 polars/matplotlib/tkinter/_tkinter.pyd/_tcl_data/tcl86t.dll **全部不存在**（连带 `pyi_rth__tkinter` 钩子不注入 → 无启动崩溃）；③ 重建 `lada-cli.exe --version` 正常启动、跑真实去码完整导出整段 test_video（3576 帧，exit 0）；④ 构建后清理为 no-op（excludes 已删干净）。BUILD_EXIT_CODE=0，体积 6.9G。**注意 PowerShell 坑**：`uv run pyinstaller` 的 INFO 走 stderr，PS5.1 把原生命令 stderr 包成 NativeCommandError，叠加 `$ErrorActionPreference=Stop` 会让构建首行即中断 —— 复刻 `Create-EXE` 调试时用 bash 跑 `venv_release_win/Scripts/python.exe -m PyInstaller ... > log 2>&1` 最稳。
- **教训**：① PyInstaller `excludes` 在干净构建里可靠生效（上面已验），但**旧 dist 可能保留构建前的包**（polars 一度让人误判）—— 看体积要认准「重新构建后」的 dist，别拿旧 artifact 下结论；体积关键且无启动钩子的 leaf 包再叠加构建后 `Remove-Item` 兜底零成本。有启动钩子的（tkinter）反过来**只能**靠 excludes（构建后删 `_tcl_data` 会触发 `pyi_rth__tkinter` 启动崩）。② cuDNN 子库严格加载、无优雅回退，删错直接 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`，验证机务必无系统 CUDA toolkit（否则 PATH 同名 DLL 会掩盖真实依赖）。
- **裁剪基本到顶**：剩余大块全是 🔴 地基（torch 核心 CUDA 1.7G、cuDNN 引擎库 690M、tensorrt_libs 2.2G、ffmpeg/ffprobe 284M）。仅剩 scipy(52M+20M) 需改加载/注册路径代码才能动（搁置）。除非换更小的依赖方案，否则 6.9G 已接近 nvidia+TRT 包的下界。



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
