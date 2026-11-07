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
- `cargo test --locked --no-default-features` 和 `cargo test --locked --no-default-features --features openvino` 均通过。CUDA 与 MLX 的运行基准来自仓库历史报告，本次同机测试使用 Rust CPU、OpenVINO CPU 和 Intel 核显。NPU 复测见下节。

## 同机实测

2026-11-07，Intel Core Ultra 7 258V、Intel Arc 核显、Manjaro Linux 7.2.9，电池供电。
三组使用同一个 release 二进制、同一份权重和 `--blk 1600`，两条 CPU 路径均指定 `--threads 8`。
测试含 8 段真实语音和 1 段由它们拼接的长音频，总计 107.25 秒。每个后端使用一个进程，
先跑一遍预热，再跑三遍，取整批解码时间中位数。热态时间含 fbank、声学推理、分块和 CTC。
“进程墙钟”含加载、预热和三遍测量。

| 后端 | 5.59 s 短音频热态 | 54.62 s 长音频热态 | 热态整批 | 热态吞吐 | 对 Rust CPU | 进程墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| Rust CPU | 233 ms | 3,169 ms | 5,495 ms | 19.5× 实时 | 1.00× | 22.18 s |
| OpenVINO CPU | 105 ms | 1,322 ms | 2,429 ms | 44.2× 实时 | 2.26× | 11.56 s |
| OpenVINO GPU | 36 ms | 253 ms | 532 ms | 201.6× 实时 | 10.33× | 7.18 s |

Rust CPU、OpenVINO CPU 和 OpenVINO GPU 的模型加载分别为 9、1352 和 264 ms。
GPU 首遇 1000、500、2000 帧区间的文件耗时分别为 1659、1358 和 1606 ms，含编译及转写。
首次运行的延迟受编译影响，持续运行可复用已编译请求。

短样本 `zh.wav` 的三组结果均为“开放时间早上九点至下午五点。”全部 9 个输入中，OpenVINO CPU 有
5 条、GPU 有 6 条与 Rust CPU 完全一致。去掉标点、空白并统一大小写后，两条 OpenVINO 路径各有
4 次字符编辑差异（Rust CPU 参照共 432 字符，0.93%）。这是后端一致性测量，语料未配人工标注。

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
Compiler-In-Plugin 和维度上界，当时的预览覆盖以视觉模型为主。本模型复用 GPU 的固定长度和掩码，
每次推理只取真实音频帧对应的 token。[OpenVINO NPU 设备文档](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

```bash
export WTL_OV_CACHE_DIR="$PWD/target/local/npu-cache"
./target/release/wasr models/wtmodel --openvino --ov-device NPU bench-out/audio/zh.wav
python3 scripts/bench_openvino.py --model models/wtmodel --npu \
  --output bench-out/openvino-npu-comparison.json \
  bench-out/audio/{0,1,2,3-sichuan,4-tianjin,5-henan,6-zh-en,zh,joined}.wav
```

初次使用某个窗口长度时编译，随后优先复用进程内请求。指定 `WTL_OV_CACHE_DIR` 后，后续进程也可
复用 OpenVINO 的编译缓存。缓存由当前运行时生成，保存在本地忽略目录。NPU 使用默认的效率策略。

同一台 Core Ultra 7 258V 上，使用现有 OpenVINO 2026.4.1 和 Linux NPU 驱动 1.38.0，
以相同 9 条输入、`--blk 1600 --threads 8` 复测。NPU 忙时计数从 8,002,455 µs
增至 9,061,794 µs。下表为预热后中位数。

| 后端 | 5.59 秒样本 | 54.62 秒样本 | 107.25 秒整批吞吐 |
|---|---:|---:|---:|
| Rust CPU | 240 ms | 3,318 ms | 19.0× 实时 |
| OpenVINO CPU | 115 ms | 1,195 ms | 49.6× 实时 |
| OpenVINO NPU | 18 ms | 115 ms | 422.2× 实时 |

NPU 热态吞吐为同轮 Rust CPU 的 22.21×、OpenVINO CPU 的 8.51×。9 个输入中 6 条与 Rust CPU
完全一致，规范化后的字符编辑差异为 3/432（0.69%）。短句、静音、极短输入以及长音频的
JSON/TXT 输出均通过验证。

空缓存下，触发 500、1000、2000 帧编译的首个文件分别用时 18.30、23.63 和 30.00 秒，包含编译和推理。
缓存占 423 MiB。复用缓存的新进程中，模型初始化 243 ms，首个 5.59 秒样本含缓存导入和转写 207 ms。

### 电池供电下的能耗

在同一台机器上用 54.62 秒音频做连续热态转写，按 OpenVINO CPU、NPU、NPU、CPU 顺序运行。
每轮先预热一次，再从 Linux 电池 `energy_now` 记录区间端点。每个 CPU 区间处理 24 条音频，
每个 NPU 区间处理 100 条，按实际完成的音频分钟归一化。整机数据包含桌面与后台进程耗电。

| 后端 | 整机耗电 / 音频分钟 | 忙碌时段平均整机功率 | 进程 CPU 时间 / 音频分钟 |
|---|---:|---:|---:|
| OpenVINO CPU | 21.42 J | 15.51–15.92 W | 9.03–9.06 s |
| OpenVINO NPU | 2.77–3.16 J | 19.80–22.43 W | 0.110–0.112 s |

在该持续负载下，NPU 因更快完成任务，每音频分钟的整机耗电约少 85–87%，忙碌区间的平均整机功率更高。
这是电池侧测量，适用于这次批转写和电源状态。原始记录见 `bench-out/openvino-npu-battery.json`。

## 四后端同轮对比

Intel Core Ultra 7 258V、Arc 核显、Intel NPU、Manjaro 7.2.9、OpenVINO 2026.4.1，
电池供电，CPU 调频策略为 `powersave`，能效偏好为 `power`。四组使用同一
`target/ov-sdk/release/wasr`、相同权重、`--blk 1600 --threads 8`。
每组先跑一遍 107.25 秒语料，再跑七遍取中位数。耗时含 fbank、推理、分块和 CTC。
冷进程的 NPU 从空 OpenVINO 编译缓存开始，核显可复用已存在的 OpenCL 驱动缓存。

| 后端 | 8 条短句的中位耗时 | 54.62 秒 | 整批热态耗时 | 热态吞吐 | RTF | 相对 Rust CPU |
|---|---:|---:|---:|---:|---:|---:|
| Rust CPU | 285 ms | 3,222 ms | 5,577 ms | 19.2× | 0.0520 | 1.00× |
| OpenVINO CPU | 130.5 ms | 1,273 ms | 2,301 ms | 46.6× | 0.0215 | 2.42× |
| OpenVINO GPU | 37 ms | 243 ms | 531 ms | 202.0× | 0.0050 | 10.50× |
| OpenVINO NPU | 16 ms | 104 ms | 227 ms | 472.5× | 0.0021 | 24.57× |

| 后端 | 七遍整批耗时范围 | 与 Rust CPU 逐字一致 | 字符编辑差异 |
|---|---:|---:|---:|
| Rust CPU | 5,409–6,105 ms | 9/9 | 0/432 |
| OpenVINO CPU | 2,229–2,606 ms | 5/9 | 4/432 |
| OpenVINO GPU | 509–572 ms | 6/9 | 4/432 |
| OpenVINO NPU | 222–248 ms | 6/9 | 3/432 |

短句为 4.53–8.10 秒，先对每条的七次耗时取中位数，再对八条结果取中位数。
字符差异的参考是 Rust CPU 的输出，比较前去掉标点和空白并统一大小写。
CUDA 和 MLX 的历史实测来自不同硬件和不同语料，见 README 和相应报告。

```bash
export WTL_OV_CACHE_DIR="$(mktemp -d "$PWD/target/local/npu-comparison-cache.XXXXXX")"
python3 scripts/bench_openvino.py --binary target/ov-sdk/release/wasr \
  --model models/wtmodel --gpu --npu --repeats 7 \
  --output bench-out/full-backend-comparison.json \
  bench-out/audio/{0,1,2,3-sichuan,4-tianjin,5-henan,6-zh-en,zh,joined}.wav
```

需要沿用上文的进程级 OpenCL 环境。JSON 含逐样本转写、首次和热态耗时、每遍耗时、整进程墙钟、主机 CPU 时间和进程峰值 RSS。

### 新进程首句与编译开销

每个后端启动三个新进程，各转写同一条 5.59 秒语音。下表取中位数。
进程墙钟从启动到退出，含模型准备、读取 WAV、输出 JSON 以及销毁上下文。
NPU 的三个新进程均命中磁盘缓存，GPU 与 CPU 未配置 OpenVINO 磁盘缓存。

| 后端 | 模型加载 | 首句解码 | 新进程总墙钟 | 首句进程峰值 RSS |
|---|---:|---:|---:|---:|
| Rust CPU | 9 ms | 252 ms | 282 ms | 154 MiB |
| OpenVINO CPU | 1,373 ms | 259 ms | 1,755 ms | 534 MiB |
| OpenVINO GPU | 268 ms | 1,540 ms | 1,943 ms | 599 MiB |
| OpenVINO NPU（缓存命中） | 228 ms | 195 ms | 485 ms | 444 MiB |

同轮空缓存 NPU 的 500 / 1000 / 2000 帧首次文件耗时分别为 17.29 / 23.28 / 28.84 秒，
含编译与解码，三个缓存文件合计 423 MiB。GPU 对应首次文件为 1.35 / 1.45 / 1.79 秒。
四后端运行完整八遍语料的进程墙钟分别为 45.48 / 20.78 / 9.47 / 71.88 秒，
峰值 RSS 分别为 218 / 593 / 594 / 1,201 MiB。这里的 NPU 数据含空缓存编译阶段。

### 首句与固定档位后续优化

批转写入口不再在 `finish` 前提交短于 20 秒的分块。18 秒音频在修复前热态中位约 56 ms，
修复后约 29 ms，文本逐字一致。NPU 默认执行精度由 `INFERENCE_PRECISION_HINT=f16` 确认；
权重在图中来自 i8 常量，但当前不是默认 W8A8 推理。

`wasr --warmup` 可在 daemon 启动前编译或导入 500/1000/2000 三个固定窗口。本机缓存命中时
导入并各做一次空推理约 0.27 秒；空缓存需要 28.78 秒并生成 423 MiB。Voxtype 在检测到三个
缓存文件后，于模型加载 250 ms 后在后台执行同样的预热。三个独立进程的 NPU 集成测试中，
5.59 秒样本首句由约 99 ms 降至约 33 ms。

把固定档位改为 `WTL_NPU_BUCKETS=500,1000,1500,2000` 后，10.05 秒样本从 28 ms 降至 21 ms，
14.95 秒从 27 ms 降至 22 ms；额外缓存约 150 MiB，进程峰值 RSS 约 726 → 869 MiB，
空缓存编译约 11.6 秒，因此没有设为默认。

补测 CPU 和 GPU 的 OpenVINO 磁盘缓存，均从空缓存生成单个短句所需模型，再启动三个新进程。
OpenVINO CPU 的首句进程墙钟中位数由无缓存的 1,755 ms 降为 1,075 ms，缓存约 130 MiB。
GPU 由 1,943 ms 降为 961 ms，缓存约 159 MiB。缓存尺寸对应本次短句所需模型。

### 约 10 分钟单文件

将 54.62 秒音频重复 11 次，生成 600.87 秒单文件。每个新进程先解码 4.53、5.59 和
54.62 秒三个文件，预热用到的长度，再解码长文件一次。峰值 RSS 和主机 CPU 时间涵盖整个进程，
包括三个预热文件，设备侧分配单独计量时需使用相应驱动工具。

| 后端 | 长文件解码 | 长文件吞吐 | 进程峰值 RSS | 进程主机 CPU 时间 | 字符编辑差异 |
|---|---:|---:|---:|---:|---:|
| Rust CPU | 37.360 s | 16.1× | 471 MiB | 263.0 CPU·s | 0/2,306 |
| OpenVINO CPU | 14.611 s | 41.1× | 656 MiB | 105.9 CPU·s | 20/2,306 |
| OpenVINO GPU | 2.343 s | 256.5× | 630 MiB | 8.58 CPU·s | 15/2,306 |
| OpenVINO NPU | 1.094 s | 549.2× | 798 MiB | 1.88 CPU·s | 16/2,306 |

### 同轮整机能耗

四个后端按 Rust CPU、OpenVINO CPU、GPU、NPU 顺序测量，再倒序测一次。每个进程先预热，
然后连续处理同一条 54.62 秒 WAV 约 23–30 秒，读取电池 `energy_now` 的起止值。
`J/音频分钟` 为两轮均值，括号中为两轮范围。电池计数包含操作系统、桌面和其他后台负载。
本测试测量持续批转写的耗电，实时麦克风工作负载需要单独测量。

| 后端 | 整机耗电 / 音频分钟 | 忙碌时段平均整机功率 | 进程 CPU 时间 / 音频分钟 |
|---|---:|---:|---:|
| Rust CPU | 51.90 J（49.43–54.37） | 14.15–15.52 W | 22.82 CPU·s |
| OpenVINO CPU | 15.61 J（14.57–16.65） | 10.82–12.66 W | 8.98 CPU·s |
| OpenVINO GPU | 8.12 J（7.91–8.32） | 22.96–26.57 W | 0.389 CPU·s |
| OpenVINO NPU | 3.12 J（3.03–3.21） | 20.26–21.95 W | 0.103 CPU·s |

NPU 的每音频分钟耗电比同轮 Rust CPU、OpenVINO CPU、OpenVINO GPU 分别低 94%、80%、62%。
在持续忙碌时，OpenVINO CPU 的平均整机瓦数最低。NPU 以更短的执行时间取得最低单位音频能耗。
原始记录在 `bench-out/full-backend-battery.json`，长文件和新进程数据在同目录的
`full-backend-long.json`、`full-backend-startup.json` 和 `full-backend-cached-startup.json`。

### CUDA 和 MLX 历史记录

仓库原有报告对另一条 58.65 秒音频 `bv58.wav` 的记录如下。硬件、音频和计时条件与上面的同轮测试不同。

| 后端 | 历史测试硬件 | 58.65 秒音频 |
|---|---|---:|
| CUDA int8 | RTX 5060 Ti 16 GB | 53 ms |
| MLX fp16 | Apple M2 Pro 16 核 GPU | 143 ms |

[CUDA 报告](REPORT.md) · [MLX 报告](MLX-REPORT.md)
