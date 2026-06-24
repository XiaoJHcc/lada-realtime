# TRT 加速 BasicVSR++ 移植设计文档

> 目标读者:一个**全新会话**的执行者。本文档自包含 —— 不依赖对话历史。
> 分支:`feat/trt-basicvsrpp`(已创建)。源参考仓库:`D:/Git/jasna`(jasna,lada 的 fork,已 clone 到本地)。

## ✅ 实现状态(2026-06-24 完成移植 + 验证,代码已落地未提交)

移植**全部完成并通过验证**。已落地文件:
- `lada/trt/__init__.py`、`lada/trt/torch_tensorrt_export.py`、`lada/trt/trt_runner.py`(去掉 `_frozen`/ONNX/YOLO 部分)
- `lada/restorationpipeline/trt_engine_paths.py`(只含 BasicVSR++ 路径辅助)
- `lada/restorationpipeline/basicvsrpp_sub_engines.py`(核心,wrapper + split forward 照搬)
- `lada/restorationpipeline/basicvsrpp_trt_compilation.py`(编译策略)
- 集成:`BasicvsrppMosaicRestorer.__init__(..., split_forward=None)` + `restore()` 三分支;`load_models` 经 `_maybe_build_trt_split_forward` 装配,模块级开关 `BASICVSRPP_TRT_ENABLED`(env `LADA_BASICVSRPP_TRT`,默认 on)、上界 `BASICVSRPP_TRT_MAX_CLIP_SIZE`(env `LADA_BASICVSRPP_TRT_MAX_CLIP`,默认 180)。
- 验证脚本:`scripts/trt_basicvsrpp_validate.py`。

**实测结果**(RTX4080 / fp16 / max_clip_size=30 / 测试 clip T=16,256×256):
- 6 个引擎全部编译成功(**deform_conv2d 能被 torch_tensorrt 编 —— 最大风险点已排除**;SPyNet preprocess、upsample 动态 batch 均 OK)。编译耗时 ~872s(max_clip_size=30,opt_level=5)。
- 数值对齐:**MAE=0.0012**(闸门 < 2/255≈0.0078,PASS),max_abs=0.037,PSNR=53.75 dB。
- 吞吐:PyTorch 52fps → TRT 165fps,**3.16x**(与 jasna 整体 3.1x 一致)。
- `load_models` → `restore()` TRT 路径 OK;`LADA_BASICVSRPP_TRT=0` 回退 PyTorch 路径 OK;`scripts/trt_smoke_test.py` 回归通过。

**注意**:已编译 `_b30`(测试)与 `_b180`(生产默认上界)两套 preprocess/upsample 引擎,loop_body 静态 batch=1 两者复用。生产默认上界 180 **已预编译完毕**,首次正式运行不再阻塞。`_b180` 在 T=60 实测 **4.47x**(28.5→127fps),MAE=0.0007,PSNR 57.97dB。

**已知无害告警**(可忽略,加载成功后才打印):`Unable to import quantization/quantize op`(modelopt,只 INT8 量化用,我们走 fp16)、`TensorRT-LLM is not installed`(只 torch.distributed 多卡用)、`This version of file is deprecated. Please generate a new pt2 saved file.`(torch.export.load 觉得 pt2 归档格式偏旧 —— 引擎仍正常 deserialize 并跑出正确结果;真到 torch 升级读不了那天,引擎也会因环境变化触发自动重编,告警自消)。**唯一真风险:别手动跨机器拷 `.engine`**,每台机首跑自己编。

**已做(2026-06-25)**:
- 端到端 CLI:`test_video_10s.mp4`(280 帧 / 1080p / 29.97fps)走 TRT 路径导出,产物 280 帧、分辨率/帧率一致、音频合并正常,~10s 墙钟完成,退出码 0。
- realtime GUI:用户实跑,**至少没崩**;有一些小问题待查(用户下次细说)。
- `.gitignore` 已加规则屏蔽 `*_sub_engines/` / `*.engine` / 验证日志 / `test_video*.mp4`(引擎绑架构、不跨机分发、体积大,不入库)。

**仍待做(下次)**:
- realtime GUI 实跑发现的小问题(用户待细说)。
- 实测 realtime **难段稳态吞吐**是否因 TRT 抬升 —— 注意上面 3–4x 是 restorer **独占** GPU 的数;真实管线 detector(YOLO)争用 4080 会打折,见 [[realtime-perf-measured-deadends]]。
- 决定是否提交 `feat/trt-basicvsrpp` 分支(代码已 stage,引擎/日志/视频已排除)。

---


## 0. 一句话目标

把 jasna 用 TensorRT 把 BasicVSR++ 拆成 6 个子引擎、推理提速 ~3x 的做法,移植进本仓库(lada-realtime),为 `BasicvsrppMosaicRestorer` 增加一条**可选的 TRT 推理路径**;非 Nvidia / 未编译时**无缝回退到现有 PyTorch 路径**。

## 1. 背景与已验证的事实(不要重新验证)

- jasna 比 lada 快 ~3x 的**唯一加速来源是 TensorRT**,不是更好的模型 —— 它直接用 lada 训练的 `mosaic_restoration_1.2` 权重。
- **网络结构 100% 相同**:已 diff `lada/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py` 与 jasna 对应文件,`__init__` 里的层定义(`spynet` / `feat_extract` / `deform_align[dir]` / `backbone[dir]` / `reconstruction` / `upsample1` / `upsample2` / `conv_hr` / `conv_last`)**逐字相同**。jasna 的三个 wrapper 伸手拿的属性在 lada 模型上全部存在、同名。
- **工具链已验证可用(路径 A,无需升级 torch)**:本机 torch 2.8.0+cu128 / python 3.13.6 / RTX4080(cc 8.9)。已装 `torch-tensorrt==2.8.0`(带 `tensorrt==10.12.0.36` 等共 7 包,**未动 torch/torchvision**)。冒烟测试 `scripts/trt_smoke_test.py` 全过:dynamo 编译 + `torch.export.save/load` 往返 + 动态 batch 1→60 数值对齐(max_abs_err 0.0004,fp16 几乎无损)。
- jasna 用 torch 2.12 + tensorrt 10.16,本机用 2.8 + 10.12。**版本差异不影响**:TRT API 在 10.x 内稳定,dynamo 编译路径在 2.8 上已实测可跑。

## 2. 已定决策(本次执行的边界)

| 决策点 | 选择 | 含义 |
|---|---|---|
| 引擎 vs 热调 clip | **锁定单一上界** | 引擎按固定 `max_clip_size`(取配置上界,如 180)编一次。realtime 实际用更短 clip 时,preprocess 引擎的动态 batch 下界已是 3,upsample 下界是 1,短 clip 直接喂即可,**不重编**。`forward` 里已有 `_PREPROCESS_MIN_BATCH=3` 的 padding 逻辑处理 t<3。 |
| 集成范围 | **只做 BasicVSR++ restorer** | YOLO 检测器的 ONNX→TRT 列为**二期**(jasna 有 `yolo_tensorrt_compilation.py`,本期不动)。 |
| 执行隔离 | **新分支** `feat/trt-basicvsrpp` | 已创建。 |
| 回退 | **必须保留 PyTorch 路径** | 非 cuda / fp32 / 引擎缺失 / 编译失败 → 用现有 `self.model(inputs=...)`。Intel Arc / Mac / CPU 用户不受影响。 |

## 3. 要移植的文件清单

源在 `D:/Git/jasna/jasna/`,目标在 `lada/`。除特别说明外,改动 = 复制 + 把 import 里的 `jasna` 改成 `lada` + 加 SPDX 头。

| # | 源(jasna) | 目标(lada) | 改动 |
|---|---|---|---|
| 1 | `trt/__init__.py` | `lada/trt/__init__.py` | 复制 `get_trt_logger` / `_engine_io_names` / `_trt_dtype_to_torch`。**只保留 BasicVSR++ 需要的部分**;`compile_onnx_to_tensorrt_engine` 等 ONNX 相关函数是 YOLO/unet 用的,本期可不带(或带但不调用)。删掉 `from jasna.engine_paths import get_onnx_tensorrt_engine_path` 等 ONNX 引用。 |
| 2 | `trt/torch_tensorrt_export.py` | `lada/trt/torch_tensorrt_export.py` | 复制 `_mute_torch_tensorrt` / `get_workspace_size_bytes` / `load_torchtrt_export` / `compile_and_save_torchtrt_dynamo` / `_save_with_dynamic_shapes`。**删掉** `from jasna._frozen import patch_frozen_torch` 调用(本仓库无 `_frozen`;`patch_frozen_torch()` 是 PyInstaller 冻结环境的补丁,源码运行可跳过 —— 改为 try/except 或直接删该行)。 |
| 3 | `trt/trt_runner.py` | `lada/trt/trt_runner.py` | 直接复制,改 import。(本期 BasicVSR++ 走 `load_torchtrt_export` 的 module 形式,`TrtRunner` 不一定用得上;若用不上可不带,二期 YOLO 再说。) |
| 4 | `engine_paths.py`(部分) | `lada/restorationpipeline/trt_engine_paths.py` 或并入 `lada/trt/__init__.py` | **只摘 BasicVSR++ 相关**:`BASICVSRPP_DIRECTIONS`、`engine_system_suffix`、`engine_precision_name`、`_basicvsrpp_sub_engine_dir`、`get_basicvsrpp_sub_engine_paths`、`all_basicvsrpp_sub_engines_exist`。**不要**带 `model_weights_dir`/`is_frozen`/unet/sd15 那些(依赖 `jasna._frozen`,且与本期无关)。引擎缓存目录用 `<权重路径同目录>/<stem>_sub_engines/`(与 jasna 一致,自然落在 `model_weights/`)。 |
| 5 | `restorer/basicvsrpp_sub_engines.py` | `lada/restorationpipeline/basicvsrpp_sub_engines.py` | **核心文件(657 行)**。复制,改 import:`from lada.trt.torch_tensorrt_export import ...`、`from lada.restorationpipeline.trt_engine_paths import ...`。三个 wrapper(`_PropagateBodyWrapper`/`_SPyNetWrapper`/`_UpsampleWrapper`)和 `BasicVSRPlusPlusNetSplit` 原样照搬 —— 它们拿的属性 lada 模型都有。 |
| 6 | `restorer/basicvrspp_tenorrt_compilation.py` | `lada/restorationpipeline/basicvsrpp_trt_compilation.py` | 复制 `get_gpu_vram_gb` / `compile_mosaic_restoration_model` / `basicvsrpp_startup_policy`。改 import。 |

> **注意网络定义的 3 处 TRT-friendly 改写**:jasna 的 `basicvsr_plusplus_net.py` 相比 lada 有 3 处等价改写(`compute_flow` 的 `t==1` 早返回;SPyNet mean/std 用 `torch.tensor` 而非 `torch.Tensor`;flow 缩放用乘 scale 张量而非原地 `*=`)。这些是为让 dynamo 能 trace。**但本期编译走的是 jasna 的 wrapper(`_SPyNetWrapper` 自己重写了 SPyNet forward,`_PreprocessWrapper` 自己算 flow),不直接调用 lada `BasicVSRPlusPlusNet.compute_flow`/`forward`**。所以**大概率不需要改 lada 的网络定义** —— wrapper 绕过了那些路径。执行时先不改;若编译报 trace 错误,再对照把这 3 处搬过来。

## 4. 集成点(唯一需要"设计"而非"照搬"的地方)

### 4.1 `BasicvsrppMosaicRestorer` 加 TRT 分支

当前 [lada/restorationpipeline/basicvsrpp_mosaic_restorer.py](lada/restorationpipeline/basicvsrpp_mosaic_restorer.py):构造签名是 `__init__(self, model, device, fp16)`,持有 `self.model`,`restore()` 里调 `self.model(inputs=inference_view)`。

jasna 的等价类持有 `self._split_forward`(TRT)或 `self.model`(PyTorch),`raw_process` 里二选一。

**集成方案**:给 lada 的类加一个可选的 `split_forward` 成员:
- `__init__` 增加可选参数 `split_forward=None`。若非 None,`restore()` 走 `self._split_forward(inference_view)`(注意 jasna `_split_forward` 吃 `(N,T,C,H,W)`,与 lada `self.model(inputs=...)` 输入形状一致,确认后直接替换那一行)。
- 保留 `self.model` 作为回退/warmup 兼容。
- `restore()` 的输入预处理、输出后处理(`mul/round/clamp/permute/unbind`)**完全不变**,只替换中间那一次 forward 调用。
- `max_frames` 分批逻辑:TRT 路径下 clip 已被 `max_clip_size` 限长,`max_frames` 通常 -1;若 >0 需确认 split_forward 是否支持分批(jasna 不分批),本期可只在 `max_frames<=0` 时走 TRT,否则回退。

### 4.2 `load_models` 装配 TRT

入口 [lada/restorationpipeline/__init__.py:25-29](lada/restorationpipeline/__init__.py#L25-L29) 的 `basicvsrpp` 分支。当前:
```python
_model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
mosaic_restoration_model = BasicvsrppMosaicRestorer(_model, device, fp16)
```
改为(伪代码):
```python
_model = load_model(...)
split_forward = None
if <TRT 开关 on> and device.type == "cuda" and fp16:
    from lada.restorationpipeline.basicvsrpp_trt_compilation import basicvsrpp_startup_policy
    from lada.restorationpipeline.basicvsrpp_sub_engines import create_split_forward
    use_trt = basicvsrpp_startup_policy(
        restoration_model_path=mosaic_restoration_model_path, device=device, fp16=fp16,
        compile_basicvsrpp=True, max_clip_size=<上界>, optimization_level=5)
    if use_trt:
        split_forward = create_split_forward(_model, mosaic_restoration_model_path, device, fp16, max_clip_size=<上界>)
mosaic_restoration_model = BasicvsrppMosaicRestorer(_model, device, fp16, split_forward=split_forward)
```
- **TRT 开关**:加一个模块级常量 `BASICVSRPP_TRT_ENABLED`(类比现有 `REALTIME_FRONTIER_GATE_ENABLED`),或读 config。首版用常量,默认按需。
- **`<上界>`**:取本仓库 clip 长度的配置上界。查 `max_clip_length`/`realtime_clip_length`/`max_clip_duration` 的最大可能值(看 config 与 CLI 默认,确认一个安全上界,如 180)。
- **warmup 保留**:`load_models` 末尾对 `mosaic_restoration_model.warmup()` 的调用**不动**。warmup 会调 `restore()`,自动走新的 TRT 路径,正好把引擎首次加载/初始化成本也付清。

### 4.3 首次编译的用户体验

`basicvsrpp_startup_policy(compile_basicvsrpp=True)` 在引擎不存在时会**同步编译**(jasna 文档说 15-60 分钟,4080 上 BasicVSR++ 单独应该快得多,几分钟级)。编译时 jasna 用 `print(message)` 输出 "Compiling sub-engine 1/6..."。本仓库是 GUI + CLI:
- CLI:print 可见,可接受。
- GUI:首次编译会让模型加载卡住数分钟且无进度。**本期最小实现**:日志 + print 即可,文档标注"GUI 进度提示"为后续优化。不要在本期陷入 GUI 进度条。

## 5. 验证步骤(执行后必做)

1. **冒烟**:`scripts/trt_smoke_test.py` 仍应通过(回归,确认没破坏 torch-tensorrt 安装)。
2. **真实编译**:用真实 `mosaic_restoration_1.2` 权重跑一次 `compile_basicvsrpp_sub_engines`,确认 6 个 `.engine` 文件落到 `model_weights/<stem>_sub_engines/`。记录编译耗时。
3. **数值对齐**:同一 clip(真实裁切帧)分别走 PyTorch 路径和 TRT 路径,比对输出。fp16 下逐像素不会完全相等,但视觉应一致、PSNR 高;给一个容差(如平均绝对误差 < 2/255)。**这一步是质量闸门** —— 加速不能毁画质。
4. **吞吐**:用 `profile_basicvsrpp.py`(jasna 有,可参考移植一个)或直接计时 `restore(clip)`,对比 PyTorch vs TRT 的 forward fps。预期 restorer 独占吞吐显著上升(jasna 整体 3x,restorer 单体提升应更高)。
5. **回退路径**:强制 `BASICVSRPP_TRT_ENABLED=False` 或 device=cpu,确认走 PyTorch 路径、结果正常。
6. **端到端**:CLI 导出一小段 `test_video.mp4`,确认产物正常;再跑 realtime GUI 预览确认不崩。

## 6. 风险与注意

- **`torch.export.save` 多子图回退**:jasna 的 `_save_with_dynamic_shapes` 在 export 失败时回退 `torch.save`。preprocess 引擎(含 SPyNet 展开)是最可能触发回退的。回退本身能工作(冒烟已验证 load 兼容两种格式),但留意编译日志里的回退提示。
- **workspace_size**:jasna 用 `free * 0.95`。编译期间会吃近乎全部空闲显存。文档提醒:**编译时关掉其他占显存的程序**。
- **引擎与权重/精度/clip 上界绑定**:换模型权重、换 fp16/fp32、换 max_clip_size 上界都会改变引擎路径(命名里含这些),触发重编。这是预期行为。
- **deform_conv2d**:`_PropagateBodyWrapper` 用 `torchvision.ops.deform_conv2d`。确认本机 torchvision 0.23.0 的该算子能被 torch_tensorrt 编(jasna 能,版本接近,大概率 OK;若不行是最大的技术风险点,需单独 fallback)。**建议执行时第一个就单独编一个 loop_body 引擎验证 deform_conv2d**,通过了再编其余。
- **不要动** `lada/gui/watch/` 和 CLI 的既有行为;TRT 是 restorer 内部替换,对上层透明。

## 7. 执行顺序建议

1. 建 `lada/trt/` 包(文件 1-3)+ 引擎路径辅助(文件 4)。
2. 移植 `basicvsrpp_sub_engines.py`(文件 5)+ 编译策略(文件 6)。
3. **先单独验证**:写个小脚本,加载真实权重 → `compile_basicvsrpp_sub_engines` 只编一个 `loop_body_backward_1`(验证 deform_conv2d)→ 全编 → `create_split_forward` → 跑一个 clip 对比数值。**这步通过 = 移植成功**,后面只是接线。
4. 改 `BasicvsrppMosaicRestorer`(4.1)+ `load_models`(4.2)。
5. 跑第 5 节全部验证。
6. 提交。

## 8. 关键文件锚点速查

- 集成入口:[lada/restorationpipeline/__init__.py](lada/restorationpipeline/__init__.py#L25)
- 改造目标:[lada/restorationpipeline/basicvsrpp_mosaic_restorer.py](lada/restorationpipeline/basicvsrpp_mosaic_restorer.py)
- 网络定义(确认结构一致,大概率不改):`lada/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py`
- jasna 核心参考:`D:/Git/jasna/jasna/restorer/basicvsrpp_sub_engines.py`、`D:/Git/jasna/jasna/trt/`、`D:/Git/jasna/jasna/engine_paths.py`
- 工具链冒烟脚本:[scripts/trt_smoke_test.py](scripts/trt_smoke_test.py)
- 记忆:`jasna-trt-acceleration-reference`(见 MEMORY.md)
