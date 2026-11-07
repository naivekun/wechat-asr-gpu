# wechat-asr-gpu

微信输入法**离线 ASR 引擎**（上游 `voicekey/crates/wtlocal`，模型容器 `.xnet`）的复刻。
Rust 引擎支持 NVIDIA CUDA、Apple Silicon MLX、OpenVINO（Intel CPU / GPU / NPU）和原生 CPU，面向批量转写。

历史 CUDA / MLX 实测（各自在对应硬件上运行，`--blk 1600`）：

| 后端 | 硬件 | bv58.wav（58.65s） | 倍实时 | 1 小时音频 |
|---|---|---:|---:|---:|
| **CUDA int8** | RTX 5060 Ti 16GB | 53 ms | **1107x** | **≈3.3 s** |
| **MLX fp16** | Apple M2 Pro（16 核 GPU / 32GB） | 143 ms | **410x** | **≈8.8 s** |
| CPU（对照） | 12 核 Intel / M2 Pro | 1.9 s / 3.3 s | 30x / 18x | 2 ~ 3.4 min |

- 正确性：CUDA / CPU 的 int8 GEMM（int32 累加）与 CPU 参考实现**逐元素完全相同**，转录 CER 0~2.6%
  （差异全是同音字/单字级）；MLX 版与 CUDA 版对比 CER 1.2%，`real1.wav` **逐字一致**。
- 模型权重来自微信输入法官方语音包，本仓库不包含、不分发；MIT 只覆盖代码。

## 快速开始

```bash
# 准备模型：从官方语音包（.apk）解包
./target/release/wasr --unpack 语音包.apk ./models/wtmodel

# NVIDIA（CUDA）——需要 CUDA Toolkit ≥12 + sm_80 以上的卡
cargo build --release
./target/release/wasr ./models/wtmodel --gpu --blk 1600 a.wav b.wav

# Apple Silicon（MLX）——需要 macOS + Apple Silicon
python3 -m pip install --user mlx       # 提供 MLX 的 C++ 头文件和 libmlx.dylib，build.rs 会自动找
cargo build --release --no-default-features --features mlx
./target/release/wasr ./models/wtmodel --mlx --batch 4 a.wav b.wav

# 没有 GPU
./target/release/wasr ./models/wtmodel --cpu --threads 8 a.wav
```

stdout 是 Tab 分隔的 `名字  音频秒  解码毫秒  RTF  文本`；`--json` / `--txt DIR` 可另存。

## OpenVINO（Linux）

准备 OpenVINO C++ SDK，本项目在 2026.4.1 上验证。已有系统 SDK 时可直接构建。
使用已解压的独立 SDK 时，仅给构建命令指定 `OPENVINO_ROOT=/path/to/openvino` 即可。

```bash
cargo build --release --locked --no-default-features --features openvino

# 输入为微信输入法独立离线语音包 APK。应用安装包 wxkb_1411_32.apk 只引用该资源。
./target/release/wasr --unpack ime_asr_decoder_model.apk models/wtmodel

# 输入为 16 kHz 单声道 16-bit PCM WAV。
./target/release/wasr models/wtmodel --openvino --threads 8 --json result.json --txt transcripts a.wav
./target/release/wasr models/wtmodel --openvino --ov-device GPU a.wav

# Intel NPU，缓存编译结果供后续进程使用。
WTL_OV_CACHE_DIR="$PWD/target/local/npu-cache" \
  ./target/release/wasr models/wtmodel --openvino --ov-device NPU a.wav

# 显式预热常用固定窗口：空缓存会编译，已有缓存只导入并做一次空推理。
WTL_OV_CACHE_DIR="$PWD/target/local/npu-cache" \
  ./target/release/wasr models/wtmodel --openvino --ov-device NPU --warmup
```

`--openvino` 默认用 CPU，`--threads` 控制 OpenVINO CPU 的推理线程数。GPU 需要带 Intel GPU 插件的
SDK 和可用的 Intel OpenCL 驱动。NPU 需要 OpenVINO NPU 插件和可访问的 Intel NPU 设备。
GPU 和 NPU 首次遇到某个长度区间时编译静态模型，`WTL_OV_CACHE_DIR` 可指定持久化编译缓存。
Rust 库可在首次加载前设置 `WTL_OPENVINO=CPU`、`WTL_OPENVINO=GPU` 或 `WTL_OPENVINO=NPU`，
继续通过 `Model`、`Stream` 和 `transcribe` 调用。
`WTL_NPU_BUCKETS=500,1000,1500,2000` 可测试更细的固定长度档位；默认仍是
500/1000/2000，避免额外约 150 MiB 缓存和首次编译开销。

[OpenVINO 验收与性能报告](docs/OPENVINO-REPORT.md)包含语音包来源、测试样本、同机对比及独立 GPU 运行环境。

## 结构

```
wav ──► fbank (Rust)  ──► 卷积前端 ──► 40 × Transformer ──► 输出层 + argmax ──► CTC 折叠
        log-mel 40 维                   D=512 / 8 头 / FF=2048     10502 词表
```

- **Rust 是引擎外壳**（~1.8k 行）：fbank、`.xnet` 解析、权重打包、分块解码、CTC、CLI。
- CUDA 走 cuBLASLt int8，MLX 走 fp16 整窗图，OpenVINO 直接使用原始 int8 权重和缩放因子构建声学图，NPU 采用 FP16 推理。
- 四套后端共用 `Model` / `Stream` / CLI，以 `--gpu` / `--mlx` / `--openvino` / `--cpu` 切换。

## 为什么 Mac 上是 fp16，CUDA 上是 int8

CUDA 版 91% 的算力是一个 int8 GEMM（int8 权重 × per-row 动态 int8 激活、int32 累加），
但 Apple GPU（M1–M4）**没有 int8 矩阵单元**，MLX 的 matmul 也不收整数。所以 MLX 后端把权重一次性
反量化成 fp16（MLX 的 matmul 内部是 fp32 累加，实测相对误差 4.6e-4，比原版 int8 量化误差 4e-3
还小一个数量级）。两边都是各自硬件上能拿到的最高算力：CUDA 用 int8 张量核，Apple 用 fp16。

## 细节

- [docs/REPORT.md](docs/REPORT.md) —— CUDA 版完整实测：线程扩展、逐阶段拆解、路线选型、nsys 数据
- [docs/MLX-REPORT.md](docs/MLX-REPORT.md) —— Mac 移植：能力矩阵、数值实验、优化历程、M5 预期
- [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) —— 已落地/已回退的各种优化记录

## License

MIT。模型权重、`.xnet` 容器格式与 `dict.decoder.utf8.txt` 均来自微信输入法官方语音包，版权归腾讯所有。
