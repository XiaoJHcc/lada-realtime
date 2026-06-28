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


## 想做什么

[lada](https://github.com/ladaapp/lada) 是一个用 AI 去除马赛克的工具，原版为**离线处理**导出视频而设计，自带的预览功能**没有为实时播放做优化**，原项目如此陈述：To watch the restored video in real-time, you'll need a **powerful machine**。

但我不信 RTX 4080 还不够 powerful，所以我 fork 了 lada，主要优化 AI 任务调度和前端视频播放策略，**优先保证视频播放，再择机去除马赛克**。

## 已做优化

**实时播放：从「数据驱动」改成「时钟驱动」**

- **永不暂停**：原版的播放节奏由 AI 处理进度决定：每一帧都要等去码结果算完才显示，算不过来就暂停缓冲。本 fork 反过来，让播放永不暂停，如果该帧 AI 已准备好，就播放去码画面，否则回退到原画。

> 所以依然需要强大的显卡，要不然还是会回退的。

- **播放更顺滑**：修复了一个让画面始终有「掉帧感」的问题——此前每隔几帧就会轻微顿一下、帧的显示时长忽长忽短，现在帧的显示节奏均匀了，肉眼明显更流畅。

**任务调度：只算「马上要看的」，且提前计算未来**

- **限制前沿**：修复原有调度下检测模型（YOLO）一直往前算下去的 BUG，改成只算修复模型（BasicVSR++）下一次任务所需的一小段。
- **提前计算**：拖动进度条后，AI 直接跳过当前帧，从前方抢跑。先放一小段原片以保证实时，等播放头走到时，缓冲好的 AI 去码帧正好就绪，无缝切换。（如果你显卡很强，可以减少跳过量）
- **减小窗口**：默认使用较低的片段窗口设置，提升响应速度，但时间稳定性下降。（如果你显卡很强，可设置更高上限）
- **偷懒**：当有多个马赛克区块时，修复模型消耗会成倍增长，所以默认只修其中一个，但会导致闪烁。（如果你显卡很强，可设置更高上限）
- **预热模型**：模型加载后先空跑一次，把显卡首次初始化的一次性开销在加载阶段付掉。

**TensorRT 加速（Nvidia）**

- 思路来自 [jasna](https://github.com/Kruk2/jasna)，这位老哥也基于 lada 做了优化，但几乎是个重做方案。我参考其加速方案思路，在 lada 基础上重新实现：
- 把 BasicVSR++ 修复模型拆成 6 个 TensorRT 子引擎，修复模型独占显卡时推理实测提速 **3–4 倍**。仅 Nvidia + FP16 可用，其他情况无缝回退到 PyTorch。
- 引擎绑定 GPU 架构 + TensorRT 版本，**首次运行需在本机编译一次**（约十几分钟，只需一次）。换显卡或升级 TensorRT 会自动失活并重编。详见下方「构建与分发」。

## 构建与分发（AI 总结的）

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

> **TensorRT 加速依赖**：本 fork 的 TRT 加速所需的 `torch-tensorrt` / `tensorrt` 已声明在 `nvidia` extra 里，`uv sync --extra nvidia`（打包脚本自动执行）会一并装好，无需手动安装。`tensorrt` 运行时库（含编译器 `nvinfer_builder_resource`，约 2.2GB）会被 PyInstaller 打进分发包，供终端用户本机编译引擎用。非 Nvidia 构建不含这部分。

> **国内网络**：仓库默认走国内镜像（PyPI 清华、PyTorch 南大、TensorRT 走 NVIDIA 官方源），见下方 [网络与镜像（国内）](#网络与镜像国内)。首次 `uv sync` 要下载 ~2.2GB 的 TensorRT 运行时库，慢的话挂代理。

### 3. TensorRT 引擎不进分发包

TRT 引擎绑定具体 GPU 架构 + TensorRT 版本，**不能跨机分发**，所以分发包里**不含**编好的引擎，由终端用户在自己机器上首次编译：

- **GUI**：首次启动会弹窗引导——无 Nvidia 卡则提示走 PyTorch；单卡提示「立即构建 / 以后再说」；多卡可选用哪块卡（选定的卡同时作为推理设备）。点构建后弹窗内显示编译进度（约十几分钟，不可取消）。跳过则下次启动再提示，直到编译过一次。
- **命令行 / 安装脚本**：装好后运行一次 `lada-cli --build-trt-engines` 预热，把编译挪到安装阶段，避免首次播放/导出时卡住。

引擎编好后缓存在 `model_weights/<模型名>_sub_engines/`，文件名编入了 GPU 架构、TensorRT 版本、精度等，升级 `torch-tensorrt` 或换显卡会自动失活重编。可用环境变量 `LADA_BASICVSRPP_TRT=0` 强制关闭 TRT、只走 PyTorch。

### 从源码直接运行（开发）

不打包、直接跑源码用于开发，详见 [`CLAUDE.md`](CLAUDE.md) 的「构建与运行」与 [`docs/windows_install.md`](docs/windows_install.md)。要点：`uv venv` → `uv sync --extra nvidia` → 打 patches → 下载模型到 `model_weights/` → GUI 还需 `build_gtk/` 就位。源码版显中文需先把 `translations/*.po` 编译成 `.mo`（见 CLAUDE.md）。

### 网络与镜像（国内）

仓库 `pyproject.toml` 默认配置了国内镜像，开箱即用：

- **PyPI**：清华 `https://pypi.tuna.tsinghua.edu.cn/simple`（default 源）
- **PyTorch（cu128）**：南京大学 `https://mirrors.nju.edu.cn/pytorch/whl/cu128`
- **TensorRT 运行时库**：NVIDIA 官方源 `https://pypi.nvidia.com`（清华镜像的 `tensorrt-cu12-libs` 只有空壳 sdist，没有真正的 wheel，必须走官方源）

打包/装依赖时若仍慢（首次要下 ~2.2GB 的 TensorRT 库），需要挂代理。

> 改动镜像或代理后，`uv.lock` 可能需要 `uv lock` 重新解析。

完整的打包流程、gvsbuild 编译 GTK 的各种坑、TensorRT 打包细节，见 [`docs/windows_packaging.md`](docs/windows_packaging.md)。

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
