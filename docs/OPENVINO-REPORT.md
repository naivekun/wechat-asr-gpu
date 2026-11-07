# OpenVINO 端到端验收

## 模型来源

应用 APK `wxkb_1411_32.apk` 的 ZIP 目录中没有离线解码包、`.xnet` 或对应词表。其
`classes3.dex` 引用了 `ime_asr_decoder_model.zip`。本次从微信 CDN 下载独立离线语音包，确认
大小为 102,920,639 字节，MD5 为 `28386481eb94ff73b28ee3949c0cf690`，与仓库
`src/pack.rs` 记录一致。[微信输入法独立离线语音包](https://download.z.weixin.qq.com/publish_words/v1/ime_asr_decoder_model/20260710220526250_ime_asr_decoder_model.apk)

`wasr --unpack ime_asr_decoder_model.apk models/wtmodel` 解出下列两个文件并验证权重和词表。模型与语音包保存在本地忽略目录。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `embed_140m_kld43wh_test_78000_1_merge_4.kv.conv1d.dyQu.onnx.xnet` | 135,646,496 B | `0674250a848ce3eb91024b3c0a6eda61b91ea45b2c40e4900130995e89cb9033` |
| `dict.decoder.utf8.txt` | 129,706 B | `08a16004cfcf5b9f1668e055f3100de9b8a44f88541819def2bbd56bfd42f484` |

## 实现和测试

Rust 复用原有的 WAV、fbank、窗口切分与拼接、CTC 和输出逻辑。OpenVINO 直接从
`.xnet` 权重生成图，计算三层卷积、线性投影、40 层 Transformer 和 TopK（K=1）。
矩阵权重在图中保留 int8 常量及缩放因子。OpenVINO CPU 和 GPU 使用 FP32 推理，NPU 使用 FP16。原生 CPU 对照使用原有的动态 int8 激活量化。

- OpenVINO CPU 编译一次动态长度模型。单元测试复用同一请求，验证了 0、4、5、9、10、17 帧以及切回 5 帧时的输出数量与 token。
- Arc 核显的动态模型在 2026.4.1 插件中触发崩溃或产生异常输出。交付路径使用 500、1000、2000 帧等静态长度，保留最近三个已编译模型。补齐区域在卷积后清零，并在注意力中屏蔽。每个长度区间的第一次推理包含编译开销。
- Intel NPU 复用静态长度和掩码。显式选择 `NPU` 后，声学图直接交给 NPU 插件，Rust 继续处理 fbank 和 CTC。
- 验收覆盖独立语音包解包、应用 APK 缺少资源时的报错、真实短语音、54.62 秒跨窗口音频、批量输入、JSON/TXT、静音、极短输入、无采样及错误采样率。
- `cargo test --locked --no-default-features` 和 `cargo test --locked --no-default-features --features openvino` 均通过。CUDA 与 MLX 的运行基准来自仓库历史报告；本次同机发布复测覆盖 Rust CPU、OpenVINO CPU、Intel Arc 核显和 Intel NPU。

## 同机实测（2026-11-08 发布复测）

2026-11-08 00:01–00:08 CST，Intel Core Ultra 7 258V / Intel Arc 核显 / Intel NPU，Manjaro Linux 7.2.9，OpenVINO 2026.4.1。电池供电，AC online=0，电量 85% → 83%；CPU governor 为 `powersave`，energy preference 为 `power`，platform profile 为 `low-power`。测试期间抑制屏幕自动锁定并停止 `voxtype.service`。四组使用同一个 `target/ov-sdk/release/wasr`、同一份权重和 `--blk 1600 --threads 8`；完整元数据和 binary SHA-256 见 `bench-out/release-bench/report.json`。

测试含 8 段真实语音和 1 段由它们拼接的长音频，总计 107.25 秒。每个后端使用一个进程；OpenVINO 后端先加载固定窗口缓存，再跑一遍预热和七遍测量，取七遍整批解码时间中位数。热态时间含 fbank、声学推理、分块和 CTC；进程墙钟、峰值 RSS 和主机 CPU 时间包含模型加载、预热和七遍测量。

| 后端 | 8 条短句中位 | 5.59 s 短音频 | 54.62 s 长音频 | 热态整批 | 热态吞吐 | 相对 Rust CPU | 进程墙钟 | 峰值 RSS | 进程 CPU 时间 | 逐字一致 | 字符编辑 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rust CPU | 286.5 ms | 237 ms | 3,190 ms | 5,425 ms | 19.8× 实时 | 1.00× | 43.877 s | 220.6 MiB | 284.10 CPU·s | 9/9 | 0/432 |
| OpenVINO CPU | 116.5 ms | 101 ms | 1,173 ms | 2,113 ms | 50.8× 实时 | 2.57× | 18.378 s | 580.9 MiB | 120.22 CPU·s | 5/9 | 4/432 |
| OpenVINO GPU | 37.0 ms | 36 ms | 222 ms | 508 ms | 211.1× 实时 | 10.68× | 6.201 s | 518.2 MiB | 7.58 CPU·s | 6/9 | 4/432 |
| OpenVINO NPU | 17.5 ms | 17 ms | 110 ms | 240 ms | 446.8× 实时 | 22.60× | 3.117 s | 732.1 MiB | 2.61 CPU·s | 6/9 | 3/432 |

“8 条短句中位”先对每条的七次耗时取中位数，再对八条结果取中位数；5.59 s 是 `zh.wav`。逐字一致和字符编辑以 Rust CPU 的输出为参照，去掉标点和空白并统一大小写；参考共 432 个字符。OpenVINO CPU / GPU 各有 4 次编辑，NPU 有 3 次。该指标衡量后端一致性，不是人工标注的准确率。

## 复现

短样本来自 Sherpa 的公开测试资源。拼接按下面命令中的顺序，每条末尾添加 0.25 秒静音。
[Paraformer 样本目录](https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/tree/main/test_wavs) ·
[SenseVoice 样本目录](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/main/test_wavs)

```bash
mkdir -p bench-out/audio
samples=https://huggingface.co/csukuangfj
for name in 0 1 2 3-sichuan 4-tianjin 5-henan 6-zh-en; do
  curl -fsSL "$samples/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/test_wavs/$name.wav" \
    -o "bench-out/audio/$name.wav"
done
curl -fsSL "$samples/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/zh.wav" \
  -o bench-out/audio/zh.wav
python3 - bench-out/audio/{0,1,2,3-sichuan,4-tianjin,5-henan,6-zh-en,zh}.wav <<'PY'
import sys
import wave

with wave.open("bench-out/audio/joined.wav", "wb") as out:
    out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    for path in sys.argv[1:]:
        with wave.open(path, "rb") as src:
            out.writeframes(src.readframes(src.getnframes()) + bytes(8000))
PY

python3 scripts/bench_openvino.py --model models/wtmodel \
  --binary target/release/wasr --gpu --output bench-out/openvino-comparison.json \
  bench-out/audio/{0,1,2,3-sichuan,4-tianjin,5-henan,6-zh-en,zh,joined}.wav
```

CPU 对比可省略 `--gpu`。脚本输出每条转写、首遍与热态耗时、RTF、逐条字符编辑距离和整批吞吐。

### 本机隔离 GPU 环境

系统 OpenVINO 2026.4.1 提供 CPU 插件。本次 GPU 用已有的用户目录 OpenVINO GenAI SDK 2026.4.1
（`~/.local/opt/openvino_genai_ubuntu24_2026.4.1.0_x86_64`），并将 Intel OpenCL 26.40.40196.8、
GMM 22.10.2 和 IGC 2.42.6 的发行包仅解压到 `target/local/intel-ocl`。
将 `intel.icd` 中的路径指向该目录下的 `libigdrcl.so`，用进程级环境变量运行。
系统包、驱动注册目录和 shell 配置均保留原状。
[Intel Compute Runtime 发行包](https://github.com/intel/compute-runtime/releases/tag/26.40.40196.8) ·
[Intel IGC 发行包](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.42.6)

```bash
OPENVINO_ROOT="$HOME/.local/opt/openvino_genai_ubuntu24_2026.4.1.0_x86_64" \
  CARGO_TARGET_DIR=target/ov-sdk cargo build --release --locked --no-default-features --features openvino

local_ocl="$PWD/target/local/intel-ocl"
OCL_ICD_VENDORS="$local_ocl/icd" \
LD_LIBRARY_PATH="$local_ocl/usr/local/lib:$local_ocl/usr/lib/x86_64-linux-gnu" \
  target/ov-sdk/release/wasr models/wtmodel --openvino --ov-device GPU bench-out/audio/zh.wav
```

复跑 GPU 基准时，给基准脚本同样的 `OCL_ICD_VENDORS` 和 `LD_LIBRARY_PATH`，并将
`--binary` 设为 `target/ov-sdk/release/wasr`。

## Intel NPU 复测

原始失败出现在无上界的动态时间轴 `[1, 1, ?, 40]`。驱动编译器在首个卷积报
`Upper bounds are not specified`，随后 Level Zero 返回 `ZE_RESULT_ERROR_INVALID_ARGUMENT`，
底层信息为 `Got negative shape dim bound: '-1'`。OpenVINO 2026.4.1 的动态 NPU 路径需要
Compiler-In-Plugin 和维度上界，本模型复用 GPU 的固定长度和掩码，每次推理只取真实音频帧对应的 token。[OpenVINO NPU 设备文档](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

本次同轮 NPU 数据：

- 8 条短句中位 **17.5 ms**，5.59 秒样本 **17 ms**，54.62 秒样本 **110 ms**。
- 107.25 秒整批热态 **240 ms = 446.8× 实时**，为 Rust CPU 的 **22.60×**。
- 9 条输入中 **6/9** 与 Rust CPU 逐字一致，规范化字符编辑 **3/432（0.69%）**。
- 缓存命中的三个新进程：模型加载中位 **233 ms**，首个 5.59 秒样本 **201 ms**，进程总墙钟 **0.506 s**，峰值 RSS **443.5 MiB**。
- 空缓存下 5.59 秒样本只触发 1000 帧窗口，首次解码 **23.785 s**，进程总墙钟 **24.104 s**，生成缓存 **146.2 MiB**，峰值为 **671.6 MiB**。
- `wasr --warmup` 生成 500/1000/2000 三档：本次空缓存 **69.14 s / 423 MiB**；缓存命中后进程总墙钟约 **0.81 s**，其中 warmup 本体约 **0.48 s**。

NPU 忙时、能耗和完整原始记录见下文及 `bench-out/release-bench/report.json`。

## 四后端同轮对比

下表是同一轮八条短句和拼接长音频的逐文件热态中位耗时，单位为 ms。每格是七次测量的中位数，`joined` 是 54.62 秒跨窗口样本。

| 后端 | 0 | 1 | 2 | 3-sichuan | 4-tianjin | 5-henan | 6-zh-en | zh | joined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rust CPU | 235 | 217 | 186 | 340 | 336 | 343 | 351 | 237 | 3,190 |
| OpenVINO CPU | 101 | 98 | 89 | 132 | 136 | 136 | 148 | 101 | 1,173 |
| OpenVINO GPU | 37 | 37 | 27 | 37 | 37 | 37 | 37 | 36 | 222 |
| OpenVINO NPU | 17 | 16 | 9 | 18 | 18 | 18 | 18 | 17 | 110 |

七遍整批耗时范围：

| 后端 | 七遍范围 |
|---|---:|
| Rust CPU | 5,384–5,510 ms |
| OpenVINO CPU | 2,095–2,162 ms |
| OpenVINO GPU | 505–511 ms |
| OpenVINO NPU | 238–243 ms |

字符编辑按文件分布：OpenVINO CPU / GPU / NPU 分别为 4 / 4 / 3 个字符；GPU 的差异包含 `joined` 中的 2 个字符，NPU 的差异包含 `joined` 中的 1 个字符。CUDA 和 MLX 的历史实测来自不同硬件和不同语料，见 README 和相应报告。

```bash
python3 scripts/bench_openvino.py \
  --binary target/ov-sdk/release/wasr \
  --model models/wtmodel \
  --gpu --npu --repeats 7 \
  --cache-root target/local/release-bench/cache \
  --output bench-out/release-bench/comparison.json \
  bench-out/audio/{0,1,2,3-sichuan,4-tianjin,5-henan,6-zh-en,zh,joined}.wav
```

需要沿用上文的进程级 OpenCL 环境；GPU/NPU 先分别填充各自的 `WTL_OV_CACHE_DIR`。完整四后端矩阵、首句、长文件和能耗记录在 `bench-out/release-bench/report.json`。

### 新进程首句与编译开销

每个后端启动三个新进程，各转写同一条 5.59 秒语音；OpenVINO 三组的磁盘缓存已由 `--warmup` 预填充。下表取三个进程的中位数。进程墙钟从启动到退出，含模型准备、读取 WAV、输出 JSON 以及销毁上下文。

| 后端 | 模型加载 | 首句解码 | 新进程总墙钟 | 首句进程峰值 RSS |
|---|---:|---:|---:|---:|
| Rust CPU | 9 ms | 257 ms | 0.293 s | 154.3 MiB |
| OpenVINO CPU | 728 ms | 269 ms | 1.153 s | 661.9 MiB |
| OpenVINO GPU | 274 ms | 614 ms | 1.007 s | 489.1 MiB |
| OpenVINO NPU | 233 ms | 201 ms | 0.506 s | 443.5 MiB |

冷缓存单次启动（同一个 5.59 秒样本）：

| 后端 | 首次解码 | 进程总墙钟 | 峰值 RSS | 生成缓存 |
|---|---:|---:|---:|---:|
| OpenVINO CPU | 266 ms | 2.004 s | 541.7 MiB | 136.3 MiB |
| OpenVINO GPU | 1.693 s | 2.113 s | 590.2 MiB | 166.9 MiB |
| OpenVINO NPU | 23.785 s | 24.104 s | 671.6 MiB | 146.2 MiB（仅 1000 帧档） |

冷启动的单条 5.59 秒语音只触发 NPU 的 1000 帧窗口；三个常用窗口全部预热的开销是 **69.14 s / 423 MiB**。三档缓存命中时再次执行 `--warmup` 的进程墙钟约 **0.81 s**。

### 输入边界

| 输入 | Rust CPU | OpenVINO CPU | OpenVINO GPU | OpenVINO NPU |
|---|---|---|---|---|
| 1 秒静音 | 空文本 / 56 ms | 空文本 / 146 ms | 空文本 / 578 ms | 空文本 / 168 ms |
| 20 ms 极短输入 | 空文本 / 1 ms | 空文本 / 0 ms | 空文本 / 0 ms | 空文本 / 0 ms |
| 空 WAV | 报错退出 | 报错退出 | 报错退出 | 报错退出 |
| 8 kHz WAV | 报错退出 | 报错退出 | 报错退出 | 报错退出 |

四后端都要求 16 kHz 单声道 16-bit PCM；空文件和错误采样率在进入声学推理前被拒绝。静音与极短输入不会产生幻觉文本。

### 首句与固定档位后续优化

批转写入口不再在 `finish` 前提交短于 20 秒的分块。18 秒音频在修复前热态中位约 56 ms，
本次修复后热态中位约 36 ms，后七次稳定在约 33 ms，文本逐字一致。NPU 默认执行精度由
`INFERENCE_PRECISION_HINT=f16` 确认；权重在图中来自 i8 常量，但当前不是默认 W8A8 推理。

`wasr --warmup` 可在 daemon 启动前编译或导入 500/1000/2000 三个固定窗口。本次缓存命中时
进程墙钟约 0.81 秒，其中 warmup 本体约 0.48 秒；空缓存生成三档需要 69.14 秒并占用 423 MiB。

另行实验把固定档位改为 `WTL_NPU_BUCKETS=500,1000,1500,2000` 后，10.05 秒样本从 28 ms
降至 21 ms，14.95 秒从 27 ms 降至 22 ms；额外缓存约 150 MiB，进程峰值 RSS 约
726 → 869 MiB，空缓存编译约 11.6 秒，因此没有设为默认，也不计入本次四后端矩阵。

CPU 和 GPU 的 OpenVINO 磁盘缓存也已补测：单个 5.59 秒样本从空缓存生成模型后，
OpenVINO CPU 的新进程总墙钟由 2.004 s 降为 1.153 s，缓存约 136.3 MiB；
GPU 由 2.113 s 降为 1.007 s，缓存约 166.9 MiB。缓存尺寸对应本次短句触发的固定窗口。

### 约 10 分钟单文件

将 54.62 秒音频重复 11 次，生成 600.87 秒单文件。每个新进程先解码 4.53、5.59 和 54.62 秒三个文件，预热用到的长度，再解码长文件一次。峰值 RSS 和主机 CPU 时间涵盖整个进程，包括三个预热文件，设备侧分配单独计量时需使用相应驱动工具。

| 后端 | 长文件解码 | 长文件吞吐 | 进程峰值 RSS | 进程主机 CPU 时间 | 字符编辑差异 |
|---|---:|---:|---:|---:|---:|
| Rust CPU | 35.786 s | 16.8× | 471.4 MiB | 261.55 CPU·s | 0/2,306 |
| OpenVINO CPU | 12.924 s | 46.5× | 695.4 MiB | 102.55 CPU·s | 20/2,306 |
| OpenVINO GPU | 2.413 s | 249.0× | 528.5 MiB | 4.99 CPU·s | 15/2,306 |
| OpenVINO NPU | 1.138 s | 528.0× | 659.5 MiB | 1.73 CPU·s | 16/2,306 |

### 同轮整机能耗

四个后端按 Rust CPU、OpenVINO CPU、GPU、NPU 顺序测量，再倒序测一次。每个进程先预热，然后连续处理同一条 54.62 秒 WAV 约 23–30 秒，读取电池 `energy_now` 的起止值。`J/音频分钟` 为两轮均值，括号中为两轮范围。电池计数包含操作系统、桌面和其他后台负载；本次已停止 `voxtype.service`，屏幕锁抑制器生效。

| 后端 | 整机耗电 / 音频分钟 | 忙碌时段平均整机功率 | 进程 CPU 时间 / 音频分钟 |
|---|---:|---:|---:|
| Rust CPU | 46.96 J（44.49–49.43） | 12.47–14.19 W | 22.905 CPU·s |
| OpenVINO CPU | 16.65 J（16.65–16.65） | 12.19–12.37 W | 9.024 CPU·s |
| OpenVINO GPU | 5.83 J（5.41–6.24） | 16.70–17.43 W | 0.406 CPU·s |
| OpenVINO NPU | 2.58 J（2.49–2.67） | 16.49–17.69 W | 0.111 CPU·s |

NPU 每音频分钟耗电比同轮 Rust CPU、OpenVINO CPU、OpenVINO GPU 分别低 **94.5%、84.5%、55.7%**。NPU 两轮的驱动忙时分别为 **22.77 s 和 22.67 s**，与进程忙碌区间基本一致；CPU/GPU 不产生 NPU 忙时。原始记录在 `bench-out/release-bench/report.json`。

### CUDA 和 MLX 历史记录

仓库原有报告对另一条 58.65 秒音频 `bv58.wav` 的记录如下。硬件、音频和计时条件与上面的同轮测试不同。

| 后端 | 历史测试硬件 | 58.65 秒音频 |
|---|---|---:|
| CUDA int8 | RTX 5060 Ti 16 GB | 53 ms |
| MLX fp16 | Apple M2 Pro 16 核 GPU | 143 ms |

[CUDA 报告](REPORT.md) · [MLX 报告](MLX-REPORT.md)
