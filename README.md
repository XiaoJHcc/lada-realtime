<h1 align="center">
  <img src="assets/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada Realtime
</h1>

<p align="center">
  <em>A fork of <a href="https://codeberg.org/ladaapp/lada">lada</a> focused on true real-time mosaic removal during playback.</em>
</p>

<p align="center">
  <em>原项目 <a href="https://codeberg.org/ladaapp/lada">lada</a> 的分支，主要优化使其更适合实时播放</em>
</p>

> [!WARNING]
> **开发中 / Work in progress.**

## 想做什么 / Motivation

[lada](https://github.com/ladaapp/lada) 是一个用 AI 去除马赛克的工具，原版为**离线处理**导出视频而设计，自带的预览功能**没有为实时播放做优化**，原项目如此陈述：To watch the restored video in real-time, you'll need a **powerful machine**.

但 RTX 4080 难道还不够 powerful 吗，导出视频可以跑到 30-45 帧，明明有实时播放的潜力，但原版的缓冲速度实在难以用于观看视频。

该 Fork 并未优化模型本身，仅优化 AI 任务调度和前端视频播放策略，优先保证视频播放，再择机去除马赛克。

---

[lada](https://github.com/ladaapp/lada) is an AI mosaic-removal tool. The original is built for **offline export**, and its built-in preview is **not optimized for real-time playback** — as the upstream README puts it: To watch the restored video in real-time, you'll need a **powerful machine**.

But isn't an RTX 4080 powerful enough? Export already runs at 30–45 fps, so the potential for real-time playback is clearly there — yet the upstream's buffer-first pacing makes it impractical for actually *watching* a video.

This fork doesn't touch the models themselves. It only reworks AI task scheduling and the front-end playback strategy: keep playback going first, remove mosaics when the GPU can keep up.

## 已做优化 / What's Done

**实时播放：从「数据驱动」改成「时钟驱动」**

- **永不暂停**：原版的播放节奏由 AI 处理进度决定：每一帧都要等去码结果算完才显示，算不过来就暂停缓冲。本 fork 反过来，让播放永不暂停，如果该帧 AI 已准备好，就播放去码画面，否则回退到原画。

> 所以依然需要强大的显卡，要不然还是会回退的。

---

**Playback: data-driven → clock-driven**

- **Never pause**: Upstream's pacing is dictated by AI progress — every frame waits for its restoration to finish, and it pauses to buffer when it can't keep up. This fork inverts that: playback never stops; if the AI frame is ready it shows the restored image, otherwise it falls back to the original.

> So you still need a powerful GPU — otherwise it just keeps falling back to the original.

---

**任务调度：只算「马上要看的」，且提前计算未来**

- **提前计算**：拖动进度条后，AI 直接跳过眼前帧，从「落点往后一小段」开始。先放一小段原片糊弄过去，等播放头走到时，AI 去码帧正好就绪，无缝切换。
- **限制前沿**：修复原有调度下检测模型（YOLO）一直往前算下去的 BUG，改成只算修复模型（BasicVSR++）下一次任务所需的一小段。
- **预热模型**：模型加载后先空跑一次，把显卡首次初始化的一次性开销在加载阶段付掉。
- **减小窗口**：默认使用较低的片段窗口设置，提升响应速度，但时间稳定性下降。

---

**Scheduling: compute only what's about to be seen, and prefetch the future**

- **Prefetch**: After a seek, the AI skips the frames right at the playhead and starts a short distance ahead. The original plays for that brief gap, and by the time the playhead catches up the restored frames are ready for a seamless switch.
- **Bounded frontier**: Fixes the upstream behavior where the detector (YOLO) races ahead indefinitely; now it only processes the short window the restorer (BasicVSR++) needs for its next clip.
- **Warmup**: Run a dummy pass right after loading so the GPU's one-time initialization cost is paid during load, not on the first real frame.
- **Smaller window**: Default to a shorter clip window for faster response, at the cost of some temporal stability.

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
