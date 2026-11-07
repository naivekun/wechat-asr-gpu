# wechat-asr-gpu

微信输入法**离线 ASR 引擎**（上游 `voicekey/crates/wtlocal`，模型容器 `.xnet`）的复刻。
Rust 引擎支持 NVIDIA CUDA、Apple Silicon MLX、OpenVINO（Intel CPU / GPU / NPU）和原生 CPU，面向批量转写。

不同硬件实测（各自在对应环境运行，`--blk 1600`）：

| 后端 | 硬件 | 测试音频 | 倍实时 | 1 小时音频 |
|---|---|---:|---:|---:|
| **CUDA int8** | RTX 5060 Ti 16GB | bv58 58.65 s：53 ms | **1107x** | **≈3.3 s** |
| **MLX fp16** | Apple M2 Pro（16 核 GPU / 32GB） | bv58 58.65 s：143 ms | **410x** | **≈8.8 s** |
| **OpenVINO GPU / NPU**（FP32 / FP16） | Intel Core Ultra 7 258V | joined 54.62 s：222 / 110 ms | **246x / 497x** | **≈14.6 s / ≈7.3 s** |
| CPU（对照） | 12 核 Intel / M2 Pro | bv58 58.65 s：1.9 s / 3.3 s | 30x / 18x | 2 ~ 3.4 min |

- 正确性：CUDA / CPU 的 int8 GEMM（int32 累加）与 CPU 参考实现**逐元素完全相同**，转录 CER 0~2.6%
  （差异全是同音字/单字级）；MLX 版与 CUDA 版对比 CER 1.2%，`real1.wav` **逐字一致**。
- OpenVINO NPU 在同一 9 条语料上 6/9 与 CPU 参考逐字一致，规范化字符编辑差异为 3/432；
  600.87 秒单文件实测 GPU 2.413 秒、NPU 1.138 秒。
- OpenVINO GPU 的推理精度为 FP32，NPU 为 FP16；两者默认都不是 INT8 激活推理。
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

# OpenVINO CPU / GPU / NPU；环境准备见 docs/OPENVINO-REPORT.md
./target/release/wasr ./models/wtmodel --openvino --ov-device NPU a.wav
```

stdout 是 Tab 分隔的 `名字  音频秒  解码毫秒  RTF  文本`；`--json` / `--txt DIR` 可另存。

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
- [docs/OPENVINO-REPORT.md](docs/OPENVINO-REPORT.md) —— OpenVINO CPU / GPU / NPU 验收、部署和实测
- [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) —— 已落地/已回退的各种优化记录

## License

MIT。模型权重、`.xnet` 容器格式与 `dict.decoder.utf8.txt` 均来自微信输入法官方语音包，版权归腾讯所有。
