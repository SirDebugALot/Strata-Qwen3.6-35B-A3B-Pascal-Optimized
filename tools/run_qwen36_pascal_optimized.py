#!/usr/bin/env python3
"""Create a local optimized GTX 1050 Ti config and run the Strata web/API server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


OPTIMIZED_ENV = {
    "STRATA_CPU_LM_HEAD": "0",
    "STRATA_CPU_ISA": "avx2",
    "STRATA_EXPERT_CACHE_FILE": "",
    "STRATA_EXPERT_CONVERSION_FAST": "1",
    "CUDA_MODULE_LOADING": "EAGER",
    "STRATA_SYNC_MOE": "0",
    "STRATA_CACHE_INIT_RESERVE_MIB": "0",
    "STRATA_CACHE_BUDGET_EXTRA_MIB": "256",
    "STRATA_NATIVE_EXPERTS": "1",
    "STRATA_EVENT_MOE": "1",
    "STRATA_ASYNC_CACHE": "1",
    "STRATA_PROMPT_BATCH": "4",
    "STRATA_KV_INITIAL_TOKENS": "2048",
    "STRATA_GPU_EXPERT_REUSE": "1",
    "STRATA_MOE_FLUSH": "1",
    "STRATA_MTP_DRAFT_SUBSET": "1",
    "STRATA_MTP_SUBSET_PER_PARTITION": "128",
    "STRATA_NATIVE_DENSE_Q6": "0",
    "STRATA_MTP_LAST_ROW": "1",
    "STRATA_IQ_INLINE_ROWS": "1",
    "STRATA_IQ_COMPACT_SCALES": "0",
    "STRATA_IQ_PREFETCH": "2048",
    "STRATA_CPU_K23_REUSE": "0",
    "STRATA_DENSE_Q6_FAST_CENTER": "0",
    "STRATA_CPU_WORKER_SPINS": "200000",
    "STRATA_CPU_COMPLETION_SPINS": "-1",
    "STRATA_DENSE_Q6_REPACK": "0",
    "STRATA_CACHE_DIRECT_PINNED": "0",
    "STRATA_GPU_EXPERT_TRANSPOSE": "0",
    "STRATA_Q8K_VECTOR_PACK": "0",
    "STRATA_ROUTER_DEVICE_FENCE": "1",
    "STRATA_MOE_EARLY_GPU": "1",
    "STRATA_EVENT_MOE_GRAPH": "0",
    "STRATA_EVENT_GRAPH_BLOCKING": "0",
    "STRATA_IQ256_GATHER": "0",
    "STRATA_MISS_STREAM_EXPERTS": "0",
    "STRATA_IQ_PAIR_GU": "0",
    "STRATA_CPU_IQ3_NIBBLE": "1",
    "STRATA_ACTQ_READONLY": "0",
    "STRATA_KV_INT8": "0",
    "STRATA_EVENT_BLOCKING": "0",
    "STRATA_CACHE_REBALANCE_EVERY": "0",
    "STRATA_PREFILL_PHASE_SWITCH": "1",
    "STRATA_PREFILL_PHASE_MIN": "512",
    "STRATA_PREFILL_MMQ": "1",
    "STRATA_PREFILL_HEAD_LOAN": "1",
    "STRATA_PREFILL_SNAPSHOT_LOAN": "1",
    "STRATA_PREFILL_LAYER_MAJOR": "1",
    "STRATA_PREFILL_LAYER_TILE": "1024",
    "STRATA_PREFILL_LAYER_WINDOW": "8192",
    "STRATA_PREFILL_LAYER_ORDERED_COMBINE": "1",
    "STRATA_PREFILL_ATTN_KV_SHARED": "1",
    "STRATA_PREFILL_ATTN_AQ": "6",
    "STRATA_PREFILL_GROUP_PREFETCH": "0",
    "STRATA_PREFILL_PROFILE": "0",
    "STRATA_PREFILL_MIXER_PROFILE": "0",
    "STRATA_PREFILL_WORKSPACE_COMPACT": "1",
    "STRATA_PREFILL_DENSE_F32": "1",
    "STRATA_PREFILL_GDN_COLUMNS": "4",
    "STRATA_PREFILL_ATTN_BANK_LAYOUT": "1",
    "STRATA_PREFILL_DENSE_ALGO6": "1",
}


def checked_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise SystemExit(f"{label} not found: {value}")
    return value


def prepare_tokenizer(model: Path, requested: str | None, runtime: Path) -> Path:
    if requested:
        tokenizer = Path(requested).expanduser().resolve()
    else:
        tokenizer = runtime / "tokenizer"
        required = ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")
        if not all((tokenizer / name).is_file() for name in required):
            runtime.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"),
                 "--gguf", str(model), "--out", str(runtime)],
                check=True,
            )
    for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
        if not (tokenizer / name).is_file():
            raise SystemExit(f"tokenizer file not found: {tokenizer / name}")
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3.6-35B-A3B MTP GGUF")
    parser.add_argument("--engine", default=str(ROOT / "build-sm61" / "strata-qwen35moe.exe"))
    parser.add_argument("--tokenizer", help="existing tokenizer directory; extracted from the GGUF when omitted")
    parser.add_argument("--cuda-bin", help="CUDA runtime bin directory to prepend for the child engine")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--context", type=int, default=25000)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--config-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 to 65535")
    if args.context < 1024:
        parser.error("--context must be at least 1024")
    if args.threads < 1:
        parser.error("--threads must be positive")

    model = checked_file(args.model, "model")
    engine = checked_file(args.engine, "engine")
    local = ROOT / ".local-data"
    runtime = ROOT / "runtime"
    local.mkdir(parents=True, exist_ok=True)
    tokenizer = prepare_tokenizer(model, args.tokenizer, runtime)

    profile = ROOT / "data" / "expert-profile-qwen36.bin"
    engine_args = [
        "-m", str(model),
        "--max-context", str(args.context),
        "--prefill-chunk", "4096",
        "--ckpt-slots", "0",
        "--no-graph",
        "--vram-reserve-mib", "32",
        "--adapt-decay", "0.85",
        "--mtp-draft", "3",
        "--draft-p", "0.9",
        "--threads", str(args.threads),
        "--cache-mb", "-1",
        "--adapt-every", "4",
        "--adapt-swaps", "16",
        "--profile-weight", "0.25",
    ]
    if profile.is_file():
        engine_args += ["--expert-profile", str(profile)]
    engine_args += ["--profile-save", str(local / "expert-profile-qwen36.local.bin")]

    config = {
        "exe": str(engine),
        "args": engine_args,
        "cwd": str(ROOT),
        "tokenizer": str(tokenizer),
        "model_name": "qwen3.6-35b-a3b-strata-pascal-optimized",
        "log": str(local / "strata-qwen36-pascal-optimized.log"),
        "lib_dirs": [str(Path(args.cuda_bin).expanduser().resolve())] if args.cuda_bin else [],
        "host": args.host,
        "port": args.port,
        "fit_max_tokens": True,
        "reasoning_budget_by_effort": {"low": 64, "medium": 128},
        "env": OPTIMIZED_ENV,
    }
    config_path = local / "strata-qwen36-pascal-optimized.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.config_only:
        print(config_path)
        return 0

    command = [
        sys.executable, str(ROOT / "serve" / "server.py"),
        "--engine", "strata", "--config", str(config_path),
        "--host", args.host, "--port", str(args.port), "--fit-max-tokens",
    ]
    if not args.no_open:
        command.append("--open")
    env = os.environ.copy()
    if args.cuda_bin:
        env["PATH"] = str(Path(args.cuda_bin).expanduser().resolve()) + os.pathsep + env.get("PATH", "")
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
