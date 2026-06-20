<h1 align="center">
  <img src="assets/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada Realtime
</h1>

<p align="center">
  <em>A fork of <a href="https://codeberg.org/ladaapp/lada">lada</a> focused on true real-time mosaic removal during playback.</em>
</p>

> [!WARNING]
> **开发中 / Work in progress.** 实时预览功能尚未完成。当前仓库主要是上游 lada 的代码 + 实时改造的规划。
> 如果你只是想用稳定的去码/导出功能，请直接使用 [上游 lada](https://codeberg.org/ladaapp/lada)。

## 这个 fork 想做什么

[lada](https://codeberg.org/ladaapp/lada) 是一个用 AI 去除/恢复打码（马赛克）成人视频（JAV）的工具。它的核心设计是**离线处理**：把整个视频跑一遍模型，**导出**成一个新文件，之后用任意播放器观看。它也带一个 GUI 实时预览，但那只是个预览 —— **没有为实时播放做优化**：当显卡跟不上时，播放器会**暂停并缓冲**，凑齐足够的帧再继续。尤其在拖动进度条（seek）跳到新片段后，模型需要重新累积多帧来降低时间维度上的闪烁，于是总会出现明显卡顿。

本 fork 想把这件事反过来做 —— **做一个真正的实时预览**：

- **永不缓冲，播放进度优先（时钟驱动）**：以播放的墙上时钟为准，绝不为了等 AI 结果而停下来。
- **显卡跟得上 → 显示去码结果**；**跟不上 → 直接播原片**，待 GPU 追上后再切回去码画面。
- **自适应降级**：通过缩短时间窗口、降低推理分辨率等手段，动态地让 GPU 持续满足 ~30 帧/秒。

这个方向的可行性来自一个观察：在导出（离线）模式下，一块 RTX 4080 可以跑到约 **90 帧/秒**，远高于常见的 30fps 视频帧率。这说明**吞吐量并不是瓶颈，延迟才是** —— 瓶颈在于时间维度模型必须先看到一整段连续帧（一个 clip）才能输出，而上游用「先缓冲一大段」来掩盖这个延迟。实时改造的核心，就是把「缓冲优先」换成「进度优先 + 降级回退」。

技术细节与改造涉及的具体文件，见 [CLAUDE.md](CLAUDE.md)。

## 当前状态

- [x] 摸清上游架构与实时卡顿的根源（见 [CLAUDE.md](CLAUDE.md)）
- [ ] 时钟驱动的实时预览窗口
- [ ] AI 帧 / 原片帧的动态切换（回退路径）
- [ ] 自适应降级（clip 长度 / 分辨率 / 跳帧）

## 从源码构建

本 fork 沿用上游的构建方式（`uv` + 各平台依赖）。详见各平台安装指南：

- [Windows](docs/windows_install.md)
- [Linux](docs/linux_install.md)
- [macOS](docs/macOS_install.md)

最简路径（CLI，Windows + Nvidia，详见 Windows 指南）：

```powershell
winget install --id Gyan.FFmpeg -e --source winget
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget

uv venv
.\.venv\Scripts\Activate.ps1
uv sync --extra nvidia            # 按显卡选：nvidia / nvidia-legacy / intel / cpu

# 打补丁 + 下载 model_weights/ 里的模型权重（见 Windows 指南）

lada-cli --input <input video path>
```

GUI 还需要 GTK4/GStreamer 系统依赖（`build_gtk/`），随后用 `lada` 启动。实时预览的改造主要发生在 GUI，因此运行 GUI 是验证改动的前提。

## 与上游的关系

本仓库是 [lada](https://codeberg.org/ladaapp/lada) 的个人 fork。上游主仓托管在 **Codeberg**（GitHub 为其镜像）。原项目的功能介绍、性能预期、Flatpak/Docker/Windows 预编译包等安装方式、训练与数据集制作等内容，请以上游 README 与文档为准。

如果你想为**去码模型本身**或上游功能做贡献，请前往上游的 [Pull requests](https://codeberg.org/ladaapp/lada/pulls) 与 [Issue tracker](https://codeberg.org/ladaapp/lada/issues)。

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
