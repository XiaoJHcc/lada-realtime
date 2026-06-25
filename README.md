<h1 align="center">
  <img src="assets/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada Realtime
</h1>

<p align="center">
  <em>原项目 <a href="https://codeberg.org/ladaapp/lada">lada</a> 的分支，专注于播放过程中真正的实时去马赛克。</em>
</p>

<p align="center">
  <a href="README.en.md">English</a> ·
  <strong>中文</strong>
</p>

> [!WARNING]
> **开发中。**

## 想做什么

[lada](https://github.com/ladaapp/lada) 是一个用 AI 去除马赛克的工具，原版为**离线处理**导出视频而设计，自带的预览功能**没有为实时播放做优化**，原项目如此陈述：To watch the restored video in real-time, you'll need a **powerful machine**。

但 RTX 4080 难道还不够 powerful 吗，导出视频可以跑到 30-45 帧，明明有实时播放的潜力，但原版的缓冲速度实在难以用于观看视频。

该 Fork 并未优化模型本身，仅优化 AI 任务调度和前端视频播放策略，优先保证视频播放，再择机去除马赛克。

## 已做优化

**实时播放：从「数据驱动」改成「时钟驱动」**

- **永不暂停**：原版的播放节奏由 AI 处理进度决定：每一帧都要等去码结果算完才显示，算不过来就暂停缓冲。本 fork 反过来，让播放永不暂停，如果该帧 AI 已准备好，就播放去码画面，否则回退到原画。

> 所以依然需要强大的显卡，要不然还是会回退的。

**任务调度：只算「马上要看的」，且提前计算未来**

- **提前计算**：拖动进度条后，AI 直接跳过眼前帧，从「落点往后一小段」开始。先放一小段原片糊弄过去，等播放头走到时，AI 去码帧正好就绪，无缝切换。
- **限制前沿**：修复原有调度下检测模型（YOLO）一直往前算下去的 BUG，改成只算修复模型（BasicVSR++）下一次任务所需的一小段。
- **预热模型**：模型加载后先空跑一次，把显卡首次初始化的一次性开销在加载阶段付掉。
- **减小窗口**：默认使用较低的片段窗口设置，提升响应速度，但时间稳定性下降。

**TensorRT 加速（Nvidia）**

- 把 BasicVSR++ 修复模型拆成 6 个 TensorRT 子引擎，修复模型独占显卡时推理实测提速 **3–4 倍**。仅 Nvidia + FP16 可用，其他情况无缝回退到 PyTorch。
- 引擎绑定 GPU 架构 + TensorRT 版本，**首次运行需在本机编译一次**（约十几分钟，只需一次）。换显卡或升级 TensorRT 会自动失活并重编。详见下方「构建与分发」。

## 构建与分发

从拉取源码到打包成可分发软件（Windows，Nvidia）。其余平台见上游 [`docs/`](docs/)。

### 1. 拉取与系统依赖

```powershell
git clone <本仓库地址> lada-realtime
cd lada-realtime
```

打包脚本会自动安装系统依赖（FFmpeg / uv / 7zip / MSYS2 / VS Build Tools 等），无需手动准备。

### 2. 一键打包

打包脚本 [`packaging/windows/package_executable.ps1`](packaging/windows/package_executable.ps1) 是端到端自动化——装系统依赖、用 gvsbuild 编译 GTK、编译翻译、下载模型权重、创建 venv、安装 Python 依赖并打补丁、PyInstaller 打 EXE、最后打成 7z 压缩包：

```powershell
# 在项目根目录运行（默认 Nvidia）
.\packaging\windows\package_executable.ps1 -extra nvidia
```

产物为 `lada.exe`（GUI）+ `lada-cli.exe`（命令行），打进 7z 分发包。常用参数：

- `-cliOnly`：只打命令行版，跳过 GTK 编译。
- `-skipWinget` / `-skipGvsbuild`：已装过系统依赖 / 已编过 GTK 时跳过，省时间。
- `-extra intel`：Intel Arc。

> **TensorRT 加速依赖**：本 fork 的 TRT 加速需要 `torch-tensorrt`。它不在默认依赖里，打包前在 venv 内 `uv pip install torch-tensorrt`（与本机 torch 版本对应，如 `torch-tensorrt==2.8.0`）。不装也能打包，只是运行时只走 PyTorch 路径。

### 3. TensorRT 引擎不进分发包

TRT 引擎绑定具体 GPU 架构 + TensorRT 版本，**不能跨机分发**，所以分发包里**不含**编好的引擎，由终端用户在自己机器上首次编译：

- **GUI**：首次启动会弹窗引导——无 Nvidia 卡则提示走 PyTorch；单卡提示「立即构建 / 以后再说」；多卡可选用哪块卡（选定的卡同时作为推理设备）。点构建后弹窗内显示编译进度（约十几分钟，不可取消）。跳过则下次启动再提示，直到编译过一次。
- **命令行 / 安装脚本**：装好后运行一次 `lada-cli --build-trt-engines` 预热，把编译挪到安装阶段，避免首次播放/导出时卡住。

引擎编好后缓存在 `model_weights/<模型名>_sub_engines/`，文件名编入了 GPU 架构、TensorRT 版本、精度等，升级 `torch-tensorrt` 或换显卡会自动失活重编。可用环境变量 `LADA_BASICVSRPP_TRT=0` 强制关闭 TRT、只走 PyTorch。

### 从源码直接运行（开发）

不打包、直接跑源码用于开发，详见 [`CLAUDE.md`](CLAUDE.md) 的「构建与运行」与 [`docs/windows_install.md`](docs/windows_install.md)。要点：`uv venv` → `uv sync --extra nvidia` → 打 patches → 下载模型到 `model_weights/` → GUI 还需 `build_gtk/` 就位。源码版显中文需先把 `translations/*.po` 编译成 `.mo`（见 CLAUDE.md）。

## License

源代码与模型沿用上游，基于 **AGPL-3.0** 授权。完整条款见 [LICENSE.md](LICENSE.md)。本 fork 的修改同样以 AGPL-3.0 发布。

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
