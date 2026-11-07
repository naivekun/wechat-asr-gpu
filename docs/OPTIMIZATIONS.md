# 优化记录（已落地 / 已回退 / 剩余）

README 只放结论，这里是取舍和实测依据。CUDA 侧的完整报告见 [REPORT.md](REPORT.md)，
Mac 侧见 [MLX-REPORT.md](MLX-REPORT.md)。

## CUDA（NVIDIA）

| 优化 | 结果 | 说明 |
|---|---|---|
| 整窗 CUDA graph | ✅ | 把 linear + 40 层 + 输出层 + argmax 约 806 个 kernel 捕获成一张图，同 t1 的窗口直接 replay（host 入队 3ms/窗口 → ~1.5µs）；long593 -6%，输出逐字一致。只对总特征帧 ≥20000（200s）的音频启用，短音频捕获开销 > 省下的时间。`WTL_GRAPH=1/0` 强制开/关 |
| fbank 并行化 | ✅ | 批模式整段喂入时按帧并行算 FFT/mel；滚动均值仍按原顺序累加，与单线程逐位一致。58.65s 音频的 fbank 29ms → ~4ms |
| cuBLASLt 算法选择缓存 | ✅ | 每形状几十 ms 的候选实测结果持久化到 `~/.cache/wasr/lt_algo.bin`（记录 GPU 名 / driver / CUDA 版本做失效保护）。单条 bv58 143ms → 120ms |
| conv2/conv3 shared-memory tiling | ❌ 回退 | bv58 121→131ms、long593 555→654ms。卷积是 latency-bound 不是吞吐-bound，shared 用量把 occupancy 压到 ~2 block/SM，反而不如"每输出一线程" |
| 剩余候选 | — | ①norm/quant 并行归约（4 个融合算子占窗口 13%，均值/方差现在 thread0 串行累加，≈-6%）②attention 换 fp16 / TileLang flash-attn（cuBLAS 版 18.8µs/次，TileLang 实测 7.15µs，C ABI 未验收）③fbank GPU 化（并行化后只剩 2~6%） |

单窗口（t=150）拆解约 7ms：attention 0.75、conv2+conv3 1.3、norm/quant 类 1.3、int8 GEMM GPU 0.9、
GEMM host 入队 0.0015（graph 之后）、epilogue+gelu 1.0（单位 ms）。

## Apple Silicon（MLX）

| 优化 | 结果 | 说明 |
|---|---|---|
| 批窗口 | ✅ | 这是 CTC 编码器，窗口之间没有状态依赖（上下文只来自音频 prev/post 重叠），`--batch N` 把 N 个固定网格窗口堆成一个 batch：GEMM 从 (390,512) 变成 (N·390,512)，fp16 利用率 11% → 45%（bv58 254ms → 191ms） |
| `mx.compile` 整窗图 | ✅ | 40 层约 480 次 dispatch 融合成一张图，按 (B,t1) 缓存，-29%（191ms → 141ms）。注意 t1 必须显式传入，不能从 t 反推（`(t1-1)*5+1 ≠ t`，会丢余数） |
| q/k/v 融合 + 用 `fast::scaled_dot_product_attention` | ✅ | attention 占窗口 14%，SDPA 在 t=390 也有 4.26 TFLOPS |
| 加大窗口到 t1≈1183（整文件一次解） | ❌ 回退 | 更快（299x）但**文本退化**（汤屋→汤污、抗震→抗奋）：超出模型训练时的窗口长度 |
| `quantized_matmul`（int8 权重） | ❌ | 比 fp16 GEMM 慢 11~24%：Apple 没有 int8 算力，反量化纯属额外开销 |
| M5+ 的 Metal 4 TensorOps | 待硬件 | M5 起才有原生 int8 `matmul2d`，届时可以 1:1 平移 CUDA 版的 W8A8 契约（也是 Mac 上唯一能"逐字一致"的路），见 MLX-REPORT.md 第五节 |

## OpenVINO / Intel NPU

当前交付路径使用 `INFERENCE_PRECISION_HINT=f16`。XNET 里的矩阵先按 i8 权重读出，再乘
per-tensor scale 变成 f32 常量；NPU 插件按 f16 执行编译后的图，不是默认 W8A8 INT8 推理。
`WTL_NPU_PROFILE=1` 会打印 `requested_precision=f16`、驱动/编译器版本和 NPU 调优属性。

| 优化 | 结果 | 结论 |
|---|---|---|
| 批入口去除重复推理 | 18 秒音频约 56 ms → 36 ms；稳定后约 33 ms，逐字一致 | ✅ 默认启用。`transcribe_batch` 在 `WHOLE_MAX` 内只取一次特征，`finish` 整段推理一次；流式 `push` 分块保持不变 |
| 后台导入并预热常用窗口 | 缓存命中时三档进程墙钟约 0.81 秒；Voxtype 集成首句约 99 → 33 ms | ✅ daemon 在启动后延迟 250 ms，在后台导入 500/1000/2000；空缓存编译留给 `wasr --warmup`。Voxtype 数字来自早期端到端测试 |
| 增加 1500 帧中间档 | 10.05 秒 28 ms → 21 ms；14.95 秒 27 ms → 22 ms | 可选：`WTL_NPU_BUCKETS=500,1000,1500,2000`。额外约 150 MiB 磁盘缓存，进程峰值 RSS 约 726 → 869 MiB，空缓存首次编译约 11.6 秒，未设为默认 |

评估过但没有合入的路径：早期 W8A8 激活量化实验有约 15% 的推理收益，但 9 条语料
多出 12 个字符编辑；QDQ optimization 反而更慢；QKV 投影合并的总吞吐没有改善，缓存和
常驻内存却分别增加约 21% 和 7%。这些实验分支未计入 2026-11-08 发布矩阵。
NPU profiling 显示剩余热点主要是
`ScaledDotProductAttention`、`MatMul` 和 `Gelu`。

显式预热空缓存：

```bash
WTL_OV_CACHE_DIR=/path/to/npu-cache \
  wasr /path/to/models/wtmodel --openvino --ov-device NPU --warmup
```

本机 Core Ultra 7 258V 上，空缓存预热 500/1000/2000 三档耗时 69.14 秒，生成 423 MiB 缓存；
再次运行命中缓存时进程墙钟约 0.81 秒，其中 warmup 本体约 0.48 秒。
