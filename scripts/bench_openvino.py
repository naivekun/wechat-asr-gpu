#!/usr/bin/env python3
"""在同一个 wasr 二进制上对比 Rust CPU、OpenVINO CPU 和可选的 OpenVINO GPU / NPU。"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from pathlib import Path


def normalize(text):
    return "".join(
        ch for ch in text.casefold()
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, actual in enumerate(hyp, 1):
            diagonal, previous = previous, row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, diagonal + (expected != actual))
    return row[-1]


def run(args, label, flags, durations, out_dir):
    output = out_dir / f"{label}.json"
    command = [
        str(args.binary.resolve()), str(args.model), *flags,
        "--threads", str(args.threads), "--blk", str(args.blk), "--json", str(output),
        *[str(path) for path in args.wavs] * (args.repeats + 1),
    ]
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("WTL_") or (k == "WTL_OV_CACHE_DIR" and label == "openvino-npu")
    }
    with tempfile.TemporaryFile(mode="w+t") as stderr:
        start = time.perf_counter()
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, env=env) as child:
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        process_s = time.perf_counter() - start
        stderr.seek(0)
        diagnostics = stderr.read()
    if child.returncode:
        raise SystemExit(f"{label} 退出码 {child.returncode}\n{diagnostics}")
    raw = json.loads(output.read_text(encoding="utf-8"))
    rows = raw["files"]
    names = [path.stem for path in args.wavs]
    n = len(names)
    if [row["name"] for row in rows] != names * (args.repeats + 1):
        raise SystemExit(f"{label} 缺少输入的转写结果\n{diagnostics}")
    expected_backend = "cpu" if label == "rust-cpu" else "openvino"
    if raw["backend"] != expected_backend or (expected_backend == "openvino" and raw["device"] != flags[-1]):
        raise SystemExit(f"{label} 后端与预期不符")

    warmed = [rows[n * i:n * (i + 1)] for i in range(1, args.repeats + 1)]
    passes_ms = [sum(row["decode_ms"] for row in group) for group in warmed]
    files = {}
    for i, name in enumerate(names):
        repeats = [group[i] for group in warmed]
        text = rows[i]["text"]
        if any(row["text"] != text for row in repeats):
            raise SystemExit(f"{label} {name} 多轮输出不一致")
        ms = statistics.median(row["decode_ms"] for row in repeats)
        files[name] = {
            "audio_s": durations[i], "first_ms": rows[i]["decode_ms"],
            "median_ms": ms, "rtf": ms / (1000 * durations[i]), "text": text,
        }
    return {
        "load_ms": raw["load_ms"], "process_s": process_s, "passes_ms": passes_ms,
        "peak_rss_mib": usage.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
        "process_cpu_s": usage.ru_utime + usage.ru_stime,
        "median_x_realtime": 1000 * sum(durations) / statistics.median(passes_ms),
        "files": files,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("target/release/wasr"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--blk", type=int, default=1600)
    ap.add_argument("--repeats", type=int, default=3, help="每个文件预热一次之后的测量次数")
    ap.add_argument("--gpu", action="store_true", help="同时测 OpenVINO GPU")
    ap.add_argument("--npu", action="store_true", help="同时测 OpenVINO NPU")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("wavs", type=Path, nargs="+")
    args = ap.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.blk < 100:
        ap.error("要求 repeats >= 1、threads >= 1、blk >= 100")
    if len({path.stem for path in args.wavs}) != len(args.wavs):
        ap.error("WAV 文件名必须互不相同")
    durations = []
    for path in args.wavs:
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2) or not wav.getnframes():
                ap.error(f"{path} 需要非空的 16 kHz 单声道 16-bit PCM WAV")
            durations.append(wav.getnframes() / 16000)
    backends = {
        "rust-cpu": ["--cpu"],
        "openvino-cpu": ["--openvino", "--ov-device", "CPU"],
    }
    if args.gpu:
        backends["openvino-gpu"] = ["--openvino", "--ov-device", "GPU"]
    if args.npu:
        backends["openvino-npu"] = ["--openvino", "--ov-device", "NPU"]
    with tempfile.TemporaryDirectory() as temp:
        runs = {}
        for label, flags in backends.items():
            print(f"运行 {label} …", flush=True)
            runs[label] = run(args, label, flags, durations, Path(temp))

    baseline = runs["rust-cpu"]
    for label, result in runs.items():
        difference, reference_len = 0, 0
        for name, row in result["files"].items():
            reference = baseline["files"][name]
            ref_text, hyp_text = normalize(reference["text"]), normalize(row["text"])
            row["char_edits_vs_rust_cpu"] = edits(ref_text, hyp_text)
            row["exact_match"] = row["text"] == reference["text"]
            difference += row["char_edits_vs_rust_cpu"]
            reference_len += len(ref_text)
        result["char_error_rate_vs_rust_cpu"] = difference / reference_len if reference_len else None
        result["speedup_vs_rust_cpu"] = result["median_x_realtime"] / baseline["median_x_realtime"]
        print(f"{label:13s}  {result['median_x_realtime']:6.1f}x 实时  "
              f"{result['speedup_vs_rust_cpu']:5.2f}x Rust CPU  加载 {result['load_ms']} ms  "
              f"字符差异 {difference}/{reference_len}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "threads": args.threads, "blk": args.blk, "repeats": args.repeats,
        "total_audio_s": sum(durations), "backends": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
