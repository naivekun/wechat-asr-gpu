# 微信 ASR 与 Voxtype 同机对比

## 测量范围

Intel Core Ultra 7 258V、Manjaro Linux 7.2.9、OpenVINO 2026.4.1，电池供电。
微信 ASR 使用官方离线 XNET 140M CTC 模型和 OpenVINO NPU FP16 推理。
Voxtype 1.1.1 沿用本机已有的 Whisper small-int8 模型、OpenVINO NPU 和中文语言设置。
这是两个实际转写方案的同机对比，模型结构、参数量和解码策略各异。

- 新进程启动、空编译缓存和输入边界行为使用本机安装的 `/usr/bin/voxtype transcribe`。
- Voxtype 的文件命令一次只读一个文件。热态、长文件和能耗使用从 v1.1.1 源码构建的测试入口，
  复用 Voxtype 自己的 WAV 转换、VAD、转写和文本处理代码，每个进程创建一次转写器。
- Voxtype 两个入口均在无网络、配置和模型只读的沙箱内运行。编译缓存复制到项目目录，
  空缓存测试使用新的独立目录。现有 daemon 保持运行。每组检查 NPU 忙时计数及回退日志。
- 测试音频沿用上轮 8 条短句（4.53–8.10 秒）、54.62 秒拼接语音和 600.87 秒连续单文件。
  热态用 107.25 秒语料先预热一遍，再跑七遍取中位数。
- 电池能耗为整机 `energy_now` 的端点差，包含桌面和后台功耗。以同一 54.62 秒文件重复转写，
  先预热，再按微信、Voxtype、Voxtype、微信顺序测量，按完成音频分钟归一化。
  WAV 文件测试覆盖转写链路，麦克风采集、热键、实时增量输出和粘贴延迟另有计量口径。

[Voxtype v1.1.1 文件入口](https://github.com/peteonrails/voxtype/blob/v1.1.1/src/app/transcribe_file.rs) ·
[Voxtype v1.1.1 OpenVINO 转写器](https://github.com/peteonrails/voxtype/blob/v1.1.1/src/transcribe/openvino_whisper.rs) ·
[微信 ASR 四后端基准](OPENVINO-REPORT.md#四后端同轮对比)

## 短句、长文件和热态吞吐

| 项目 | 微信 ASR NPU | Voxtype NPU | 微信用时比 / 吞吐比 |
|---|---:|---:|---:|
| 8 条短句的中位耗时 | 17.5 ms | 278.2 ms | 15.9× 加速 |
| 5.59 秒普通话短句 | 17 ms | 185.1 ms | 10.9× 加速 |
| 54.62 秒多段语音 | 111 ms | 2,214.5 ms | 20.0× 加速 |
| 600.87 秒连续单文件 | 1.125 s | 25.014 s | 22.2× 加速 |
| 107.25 秒整批热态吞吐 | 443.2× 实时 | 24.3× 实时 | 18.2× 吞吐 |
| 整批 RTF | 0.00226 | 0.04111 | |
| 整批七遍耗时范围 | 237–245 ms | 4,294–4,525 ms | |
| 600.87 秒测试进程峰值 RSS | 797 MiB | 2,512 MiB | |
| 600.87 秒测试进程主机 CPU 时间 | 1.87 CPU·s | 14.97 CPU·s | |

短句中位数先按每条七遍取中位数，再对八条结果取中位数。
长文件的进程峰值 RSS 和主机 CPU 时间涵盖三个预热文件、模型加载和长文件，
为 Linux 进程侧指标。七轮中每个后端各样本的转写均稳定。

| 语音 | 时长 | 微信 ASR NPU | Voxtype NPU |
|---|---:|---:|---:|
| Paraformer 0 | 5.61 s | 17 ms | 288 ms |
| Paraformer 1 | 5.16 s | 17 ms | 284 ms |
| Paraformer 2 | 4.53 s | 10 ms | 234 ms |
| 四川 | 7.84 s | 18 ms | 272 ms |
| 天津 | 7.81 s | 18 ms | 364 ms |
| 河南 | 7.99 s | 18 ms | 362 ms |
| 中英混合 | 8.10 s | 18 ms | 213 ms |
| SenseVoice 普通话 | 5.59 s | 17 ms | 185 ms |

上一轮同机同语料中，微信 Rust CPU、OpenVINO CPU、Arc 核显分别为
19.2×、46.6×、202.0× 实时，供对照。本轮直接对比两者的 NPU 路径。

## 新进程首句

均转写同一条 5.59 秒普通话样本。命中磁盘编译缓存，每个后端启动三次。
墙钟包括初始化、读取音频、生成转写和进程退出。

| 后端 | 进程墙钟中位数 | 三次范围 | 峰值 RSS 中位数 |
|---|---:|---:|---:|
| 微信 ASR NPU | 504 ms | 495–526 ms | 444 MiB |
| Voxtype NPU 安装版 | 1,506 ms | 1,487–1,799 ms | 2,521 MiB |

空磁盘编译缓存的首句各测一次，音频与上表一致。

| 后端 | 空缓存首句墙钟 | 该次生成的编译缓存 | 峰值 RSS |
|---|---:|---:|---:|
| 微信 ASR NPU | 23.06 s | 139 MiB | 677 MiB |
| Voxtype NPU 安装版 | 55.70 s | 850 MiB | 2,551 MiB |

微信 ASR 按窗口尺寸编译，本次首句只触发 1,000 帧模型。微信 ASR 的三个常用长度
都预编译时占约 423 MiB。Voxtype 本次生成两个编译缓存文件。

## 电池能耗

| 后端 | 两轮平均耗电／音频分钟 | 两轮范围 | 忙碌区间平均整机功率 | 进程 CPU 时间／音频分钟 |
|---|---:|---:|---:|---:|
| 微信 ASR NPU | 2.37 J | 2.19–2.55 J | 14.37–17.05 W | 0.102 CPU·s |
| Voxtype NPU | 39.54 J | 35.95–43.14 J | 14.62–17.33 W | 1.228 CPU·s |

微信每轮处理 217 条，Voxtype 每轮处理 11 条，使两者的测量区间接近 25–30 秒。
整机功率区间重叠，微信 ASR 更快完成同样的音频量，单位音频耗电约少 94%。
该数值来自持续 WAV 批转写。每个测量区间内两个方案的 NPU 忙时计数均增加。

## 输入边界

| 样本 | 微信 ASR NPU | Voxtype NPU（安装版） |
|---|---|---|
| 1 秒静音 | 空文本 | 输出「你不要再說了」 |
| 20 ms 静音 | 空文本 | 输出「你不要再說了」 |
| 空 WAV | 报错 | 报错 |
| 8 kHz 静音 | 报错，需要 16 kHz PCM | 自动重采样后输出「你不要再說了」 |

Voxtype 的结果来自现有配置和本次四个样本。两者对 5.59 秒的已知句子识别内容一致，
Voxtype 写作「开放时间早上9点至下午5点。」，微信 ASR 使用中文数字。

## 转写差异

两边的输出均以原 Rust CPU 微信模型为同一参照，去掉标点和空白并统一大小写。
同时使用 OpenCC 统一繁简再复算，保留用词、数字格式和增删差异。

| 后端 | 9 条逐字一致数 | 9 条字符编辑数 / 432 | 统一繁简后 / 432 | 10 分钟统一繁简后 / 2,306 |
|---|---:|---:|---:|---:|
| 微信 ASR NPU | 6/9 | 3 | 3 | 16 |
| Voxtype NPU | 0/9 | 175 | 139 | 763 |

参照与微信 ASR 使用同一模型，指标衡量与该参照的差异。语料缺少独立人工标注，
因此这组数字不用于给两个模型的准确率排序。原始逐条文本保存在跑分 JSON。

## 原始记录

- `bench-out/voxtype-comparison.json`，八遍短长语料、延迟及文本
- `bench-out/voxtype-long.json`，600.87 秒单文件、文本及进程资源
- `bench-out/voxtype-startup.json`，安装版和微信 ASR 的三次缓存命中新进程
- `bench-out/voxtype-empty-cache.json`、`bench-out/voxtype-wechat-empty-cache.json`，空缓存首句
- `bench-out/voxtype-battery.json`，四轮整机能耗
- `bench-out/voxtype-quality.json`，逐条字符编辑距离
- `bench-out/voxtype-edge-cases.json`、`bench-out/voxtype-wechat-edge.json`，输入边界

测量前后的 SHA-256 校验一致，覆盖 Voxtype 的现用配置和两个已有备份。
Voxtype daemon 始终为同一进程，系统包未改动。
