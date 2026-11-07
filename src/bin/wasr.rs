//! wasr —— 微信输入法离线 ASR 引擎（wtlocal / .xnet）的批模式 CLI。
//!
//! 用法：
//!   wasr <模型目录> [选项] <a.wav> [b.wav ...]
//!
//! 选项：
//!   --gpu            用 GPU 后端（需要 NVIDIA GPU + CUDA 编译的 libwasr_gpu.so）
//!   --openvino       用 OpenVINO Runtime（--ov-device CPU|GPU|NPU，默认 CPU）
//!   --cpu            强制 CPU 后端（对照用）
//!   --threads N      CPU 线程上限（默认 8；实测 8 是拐点）
//!   --blk N          解码块长（特征帧，10ms/帧）：默认 1600（批模式最优），流式才用 400
//!   --json FILE      逐文件结果写 JSON
//!   --txt DIR        每条音频的转录另存为 DIR/<name>.txt
//!   --warmup         只编译/导入常用 OpenVINO 静态窗口，不转写
//!   --unpack PKG DIR 解包官方语音包（.apk）到 DIR，得到模型文件
//!
//! stdout 为 Tab 分隔：`名字  音频秒  解码毫秒  RTF  文本`；stderr 打加载信息与批聚合吞吐。
use anyhow::{bail, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wechat_asr_gpu::{load, pack};

/// 后端名字（打印用）
fn backend_name(m: &wechat_asr_gpu::Model) -> &'static str {
    if m.openvino_device().is_some() {
        return "OpenVINO";
    }
    if !m.gpu_enabled() {
        return "CPU";
    }
    #[cfg(feature = "mlx")]
    if m.mlx_enabled() {
        return "MLX";
    }
    "GPU"
}

/// JSON 里的后端名
fn backend_tag(m: &wechat_asr_gpu::Model) -> &'static str {
    if m.openvino_device().is_some() {
        return "openvino";
    }
    if !m.gpu_enabled() {
        return "cpu";
    }
    #[cfg(feature = "mlx")]
    if m.mlx_enabled() {
        return "mlx";
    }
    "gpu"
}

fn usage_error(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn read_wav(path: &str) -> Result<Vec<i16>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000
        || spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int
    {
        bail!("需要 16 kHz 单声道 16-bit PCM WAV");
    }
    let pcm = reader.into_samples::<i16>().collect::<std::result::Result<Vec<_>, _>>()?;
    if pcm.is_empty() {
        bail!("WAV 没有采样");
    }
    Ok(pcm)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        eprintln!(
            "用法: wasr <模型目录> [--gpu|--mlx|--openvino|--cpu] [--ov-device CPU|GPU|NPU] [--batch N] [--threads N] [--blk N] [--json FILE] [--txt DIR] <a.wav> [b.wav ...]\n\
             --mlx 需要 macOS + 用 --features mlx 编译（见 README）；--batch 是 MLX 批窗口数（默认 4）\n\
             --openvino 需要用 --features openvino 编译；--ov-device 默认 CPU\n\
             --warmup 可单独使用，预编译 WTL_NPU_BUCKETS 或默认 500/1000/2000 帧\n\
             解包官方语音包: wasr --unpack <pak.apk> <模型输出目录>"
        );
        std::process::exit(2);
    }
    if args[0] == "--unpack" {
        let t = Instant::now();
        if args.len() != 3 {
            usage_error("用法: wasr --unpack <离线语音包.apk> <模型输出目录>");
        }
        if let Err(e) = pack::unpack(args[1].as_ref(), args[2].as_ref()) {
            eprintln!("解包失败: {e:#}");
            std::process::exit(1);
        }
        println!("unpacked in {:.1}s", t.elapsed().as_secs_f64());
        return;
    }

    let dir = PathBuf::from(&args[0]);
    let mut files: Vec<String> = Vec::new();
    let (mut json_out, mut txt_dir) = (None::<String>, None::<String>);
    let mut backend = None;
    let mut ov_device = None;
    let mut batch = 4usize;
    let mut warmup = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" | "--mlx" | "--openvino" | "--cpu" => backend = Some(args[i].as_str()),
            "--ov-device" => {
                i += 1;
                ov_device = Some(args.get(i).unwrap_or_else(|| usage_error("--ov-device 缺少设备名")));
            }
            "--batch" => {
                i += 1;
                batch = args[i].parse().unwrap_or(4);
            }
            "--threads" => {
                i += 1;
                std::env::set_var("WTL_THREADS", &args[i]);
            }
            "--blk" => {
                i += 1;
                std::env::set_var("WTL_BLK", &args[i]);
            }
            "--json" => {
                i += 1;
                json_out = Some(args[i].clone());
            }
            "--txt" => {
                i += 1;
                txt_dir = Some(args[i].clone());
            }
            "--warmup" => warmup = true,
            other => files.push(other.to_string()),
        }
        i += 1;
    }
    if ov_device.is_some() && backend != Some("--openvino") {
        usage_error("--ov-device 需要与 --openvino 一起使用");
    }
    if let Some(backend) = backend {
        for key in ["WTL_GPU", "WTL_MLX", "WTL_OPENVINO"] {
            std::env::remove_var(key);
        }
        match backend {
            "--gpu" => std::env::set_var("WTL_GPU", "1"),
            "--mlx" if cfg!(feature = "mlx") => std::env::set_var("WTL_MLX", "1"),
            "--mlx" => usage_error("这个二进制没有编 MLX 后端，请使用 --features mlx"),
            "--openvino" if cfg!(feature = "openvino") => {
                std::env::set_var("WTL_OPENVINO", ov_device.map_or("CPU", String::as_str));
            }
            "--openvino" => usage_error("这个二进制没有编 OpenVINO 后端，请使用 --features openvino"),
            _ => {}
        }
    }
    if std::env::var("WTL_BLK").is_err() {
        std::env::set_var("WTL_BLK", "1600"); // 批模式默认块长
    }

    if files.is_empty() && !warmup {
        usage_error("没有输入音频");
    }

    let t0 = Instant::now();
    let m = match load(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载模型失败: {e:#}\n（先跑 `wasr --unpack <微信语音包.apk> <dir>` 准备模型目录）");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let device = m.openvino_device().map(|d| format!("({d})")).unwrap_or_default();
    eprintln!(
        "load {load_ms:.0}ms  backend={}{device}  threads={}  blk={}  batch={}",
        backend_name(&m),
        std::env::var("WTL_THREADS").unwrap_or_else(|_| "8".into()),
        std::env::var("WTL_BLK").unwrap_or_else(|_| "1600".into()),
        batch
    );
    if warmup {
        let buckets = match std::env::var("WTL_NPU_BUCKETS") {
            Ok(raw) => raw
                .split(',')
                .map(|value| value.parse::<usize>().ok())
                .collect::<Option<Vec<_>>>()
                .unwrap_or_else(|| usage_error("WTL_NPU_BUCKETS 需要逗号分隔的正整数")),
            Err(_) => vec![500, 1000, 2000],
        };
        let start = Instant::now();
        if let Err(error) = m.warmup(&buckets) {
            eprintln!("warmup failed: {error:#}");
            std::process::exit(1);
        }
        eprintln!(
            "warmup {buckets:?} in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        wechat_asr_gpu::unload();
        return;
    }
    let mut rows = Vec::new();
    let batch_t0 = Instant::now();
    let (mut total_audio, mut total_ms) = (0.0f64, 0.0f64);
    for f in &files {
        let name = Path::new(f).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let pcm = match read_wav(f) {
            Ok(pcm) => pcm,
            Err(e) => {
                eprintln!("跳过 {f}: {e}");
                continue;
            }
        };
        let audio = pcm.len() as f64 / 16000.0;
        let t = Instant::now();
        // 批模式整段喂入：MLX 后端且 --batch>1 时走批窗口，否则走原来的串行分块
        let text = wechat_asr_gpu::transcribe(m.clone(), &pcm, batch);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total_audio += audio;
        total_ms += ms;
        println!("{name}\t{audio:.2}\t{ms:.0}\t{:.4}\t{text}", ms / 1000.0 / audio);
        if let Some(d) = &txt_dir {
            let _ = std::fs::create_dir_all(d);
            let _ = std::fs::write(Path::new(d).join(format!("{name}.txt")), &text);
        }
        rows.push((name, audio, ms, ms / 1000.0 / audio, text));
    }
    if rows.is_empty() {
        eprintln!("没有可转写的音频");
        std::process::exit(1);
    }
    let wall = batch_t0.elapsed().as_secs_f64();
    eprintln!(
        "\n批模式: {} 条, 音频 {:.1}s, 解码合计 {:.2}s, 批墙钟 {:.2}s, 聚合 {:.1}x 实时, 平均 RTF {:.4}",
        rows.len(),
        total_audio,
        total_ms / 1e3,
        wall,
        total_audio / wall,
        total_ms / 1e3 / total_audio
    );

    if let Some(p) = json_out {
        let mut f = std::fs::File::create(&p).expect("create json");
        let esc = escape_json;
        writeln!(f, "{{").unwrap();
        if let Some(device) = m.openvino_device() {
            writeln!(f, "  \"device\": \"{}\",", esc(device)).unwrap();
        }
        writeln!(
            f,
            "  \"backend\": \"{}\", \"threads\": {}, \"blk\": {}, \"batch\": {}, \"load_ms\": {:.0},",
            backend_tag(&m),
            std::env::var("WTL_THREADS").unwrap_or_else(|_| "8".into()),
            std::env::var("WTL_BLK").unwrap_or_else(|_| "1600".into()),
            batch,
            load_ms
        )
        .unwrap();
        writeln!(
            f,
            "  \"total_audio_s\": {:.2}, \"total_decode_s\": {:.2}, \"batch_wall_s\": {:.2}, \"agg_x_realtime\": {:.1},",
            total_audio,
            total_ms / 1e3,
            wall,
            total_audio / wall
        )
        .unwrap();
        writeln!(f, "  \"files\": [").unwrap();
        for (k, (name, audio, ms, rtf, text)) in rows.iter().enumerate() {
            writeln!(
                f,
                "    {{\"name\": \"{}\", \"audio_s\": {:.2}, \"decode_ms\": {:.0}, \"rtf\": {:.4}, \"text\": \"{}\"}}{}",
                esc(name),
                audio,
                ms,
                rtf,
                esc(text),
                if k + 1 == rows.len() { "" } else { "," }
            )
            .unwrap();
        }
        writeln!(f, "  ]").unwrap();
        writeln!(f, "}}").unwrap();
        eprintln!("JSON 已写入 {p}");
    }
    // 释放模型：触发出 GPU 上下文销毁（写回 cublasLt 算法缓存、释放显存）
    wechat_asr_gpu::unload();
}

fn escape_json(s: &str) -> String {
    use std::fmt::Write;
    let mut escaped = String::new();
    for ch in s.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\0'..='\u{1f}' => write!(escaped, "\\u{:04x}", ch as u32).unwrap(),
            _ => escaped.push(ch),
        }
    }
    escaped
}
