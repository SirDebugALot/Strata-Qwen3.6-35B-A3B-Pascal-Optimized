#!/usr/bin/env python3
"""Strata-Qwen36 one-click setup and start: Qwen3.6-35B-A3B (Unsloth's MTP GGUFs) on a normal PC.

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run setup.py,
                                                                which hands over to this file

The first time it asks which size and how much context, then installs everything and starts the model on
http://127.0.0.1:8081 (OpenAI- and Anthropic-compatible API, the Strata web app). Every later start skips straight to
running the model.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv
  4. compiles the engine (engines/qwen35moe) for your GPU - it installs the build tools first if needed (asks)
  5. downloads the model from Hugging Face (resumable) and checks its SHA-256
  6. writes the tokenizer (from the GGUF) next to it
  7. writes run-qwen36-<size>.bat / .sh and starts the model

This fork keeps its files apart from an upstream Strata install on the same PC: settings in %APPDATA%\\Strata-Qwen36
(~/.config/strata-qwen36), the model files in Strata-Qwen36-data next to this folder, the server on port 8081.

Options: --model Q4_K_M, --context 32768, --port 8081, --gpu N, --yes (recommended answers, no questions), --setup
(install another size / change settings instead of starting), --no-start, --host 0.0.0.0 --api-key KEY,
--data-dir DIR, --gguf FILE (a GGUF you already have), --threads N, --check (only check this PC).
Upstream's setup for Qwen3.8-Flash-Next: setup.py --flash-next.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setup as S
from setup import ROOT, WIN, ask, fail, ok, run, say, step, warn

# Every file from a fixed commit of the repository (the `sha` of https://huggingface.co/api/models/<repo>), as upstream
# pins its downloads (#214); a revision the repository no longer has falls back to its current file (S.download).
HF_REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
HF_REVISION = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"   # 2026-10-01
HF = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/"

# The sizes this engine runs (its CPU and GPU expert kernels: Q4_K, Q5_K, Q6_K experts; dense tensors of any K-quant
# or float type).  ram_gb: the PC's RAM for this size (the experts are pinned in RAM, plus Windows and the rest).
MODELS = {
    "Q4_K_M": {"file": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "bytes": 22663387424,
               "sha256": "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
               "about": "4-bit (Unsloth Dynamic), with the MTP layer", "experts_gb": 20.1, "ram_gb": 32},
}
CONTEXTS = [8192, 32768, 65536, 131072, 262144]
KV_BYTES_PER_TOKEN = 11 * 2 * 2 * 256 * 2   # (10 attention layers + the MTP layer) x K,V x 2 heads x 256 x fp16
EFFORT_BUDGETS = {"low": 1024, "medium": 4096}   # thinking tokens per reasoning level (serve/server.py)
DEFAULT_PORT = 8081                          # upstream Strata's is 8080: both can run at once
EXE = "strata-qwen35moe.exe" if WIN else "strata-qwen35moe"
ENGINE_DIR = ROOT / "engine-qwen35moe"
ENGINE_SOURCES = ("engines/qwen35moe/CMakeLists.txt", "engines/qwen35moe/src")
PROFILE = ROOT / "data" / "expert-profile-qwen36.bin"


def tag(model: str) -> str:
    return f"qwen36-{model.lower()}"


def installed_configs():
    return sorted(ROOT.glob("strata-qwen36-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


# ------------------------------------------------------------------------------------------------ the PC
def physical_cores() -> int:
    try:
        import psutil
        return max(1, psutil.cpu_count(logical=False) or 1)
    except Exception:                                   # psutil not installed yet (the first check)
        return max(1, (os.cpu_count() or 2) // 2)


def pick_gpu(want):
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC")
    if want is not None:
        g = next((x for x in found if x["index"] == want), None)
        if g is None:
            fail(f"there is no GPU {want}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        usable = [x for x in found if S.gpu_problem(x) is None] or found
        g = max(usable, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    if len(found) > 1:
        S.gpu_table(found)
        say(f"  Using {S.gpu_name(g)}; this engine runs on one GPU (another one: --gpu N).")
    return {**g, "count": len(found), "archs": [g["arch"]]}


# ------------------------------------------------------------------------------------------------ the engine
def build_engine(gpu, yes) -> Path:
    """Compile engines/qwen35moe for this GPU into engine-qwen35moe/; again only when its sources changed (a
    `git pull`) or the GPU needs code the build does not have."""
    ENGINE_DIR.mkdir(exist_ok=True)
    stamp = ENGINE_DIR / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    src = S.source_hash(ENGINE_SOURCES)
    archs = sorted({int(x) for x in gpu["archs"]} | {int(x) for x in meta.get("archs", [])})
    if (ENGINE_DIR / EXE).exists() and meta.get("src") == src and set(archs) <= set(meta.get("archs", [])):
        ok("engine already built for this PC")
        return ENGINE_DIR
    nvcc, vcvars = S.install_build_tools({**gpu, "archs": archs}, yes)
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) + " (a few minutes, once) ...")
    S.cmake_build(ROOT / "engines" / "qwen35moe", ROOT / "build-qwen35moe", "strata-qwen35moe",
                  [f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(str(x) for x in archs)}", f"-DCMAKE_CUDA_COMPILER={nvcc}"],
                  vcvars, "build-qwen35moe.bat")
    shutil.copy2(ROOT / "build-qwen35moe" / EXE, ENGINE_DIR / EXE)
    bindir = Path(nvcc).parent                          # cuBLAS comes from the toolkit that compiled it
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    version = engine_source_version()
    stamp.write_text(json.dumps({"source": "local", "version": version, "archs": archs, "cuda_dirs": dirs,
                                 "src": src}, indent=1))
    ok(f"engine compiled: {ENGINE_DIR / EXE}")
    return ENGINE_DIR


def engine_source_version() -> str:
    import re
    m = re.search(r"project\(strata_qwen35moe VERSION ([\d.]+)",
                  (ROOT / "engines" / "qwen35moe" / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


# ------------------------------------------------------------------------------------------------ the model
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def get_model(model: str, models_dir: Path) -> Path:
    d = MODELS[model]
    dst = models_dir / model / d["file"]
    S.download(HF + d["file"], dst, f"{d['file']} ({d['bytes'] / 1e9:.1f} GB)")
    checked = dst.with_name(dst.name + ".sha256")
    if not (checked.exists() and checked.read_text().strip() == d["sha256"]):
        say(f"  Checking {dst.name} (SHA-256, about a minute) ...")
        got = sha256(dst)
        if got != d["sha256"]:
            dst.with_name(dst.name + ".done").unlink(missing_ok=True)
            fail(f"{dst.name} is damaged (SHA-256 {got[:16]}..., expected {d['sha256'][:16]}...)",
                 f"delete {dst} and run setup again")
        checked.write_text(got)
    ok(f"model file checked: {dst}")
    return dst


def check_gguf(path: Path) -> None:
    """A GGUF given with --gguf: the architecture this engine runs (the engine checks every tensor when it starts)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        arch = GGUFFile(path).metadata.get("general.architecture")
    except Exception as e:
        fail(f"{path} is not a GGUF file ({e})")
    if arch != "qwen35moe":
        fail(f"{path.name} is a '{arch}' model; this setup runs Qwen3.5/3.6 MoE GGUFs (qwen35moe)")


# ------------------------------------------------------------------------------------------------ start
def start(cfg_path: Path, port: int | None, gpu: int | None = None, open_browser=True, keep=None) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], cfg["args"][cfg["args"].index("-m") + 1]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    if keep and any(cfg.get(k) != v for k, v in keep.items()):
        cfg.update(keep)
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        ok("saved for this model: " + ", ".join("api key" if k == "api_key" else f"{k} {v}" for k, v in keep.items()))
    cfg_path.touch()                                    # the most recently used model
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", DEFAULT_PORT))]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if open_browser:
        cmd.append("--open")
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', 'the model')}: it loads about 20 GB into RAM and locks it for the GPU.")
    say("  While it does, your PC can be slow for a few seconds. Closing this window stops the model.")
    say("  " + "-" * 100)
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), help="the size (only Q4_K_M for now)")
    ap.add_argument("--context", type=int, help="context length in tokens (at most 262144, the trained length)")
    ap.add_argument("--port", type=int, help=f"the server's port (default: the install's, {DEFAULT_PORT} for a new one)")
    ap.add_argument("--gpu", type=int, help="the GPU, numbered as nvidia-smi numbers them (default: the biggest)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0)")
    ap.add_argument("--data-dir", help=f"where the model files go: default {S.DATA_DIR} next to this folder")
    ap.add_argument("--gguf", help="use a Qwen3.5/3.6 MoE GGUF you already have instead of downloading one")
    ap.add_argument("--threads", type=int, help="CPU threads for the experts (default: the physical cores)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another size or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    a = ap.parse_args()
    say("Strata-Qwen36 - Qwen3.6-35B-A3B on a normal PC (a GPU + system RAM + CPU)")

    # ---- 0. already installed: just start it
    have = installed_configs()
    keep = {"host": a.host, "api_key": a.api_key}
    if have and not (a.setup or a.model or a.check or a.no_start or a.gguf):
        if len(have) == 1:
            return start(have[0], a.port, a.gpu, keep=keep)
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another size / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, a.gpu, keep=keep)

    # ---- 1. the PC
    step(1, "checking your PC")
    gpu = pick_gpu(a.gpu)
    ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, compute capability {S.cc(gpu)}, driver {gpu['driver']}")
    if int(gpu["arch"]) < 75:
        fail(f"{gpu['name']} is older than the RTX 20 series (compute capability {S.cc(gpu)}; 7.5 or newer is needed)")
    if S.driver_major(gpu) < S.MIN_DRIVER:
        fail(f"the NVIDIA driver is too old ({gpu['driver']}; {S.MIN_DRIVER} or newer is needed)",
             "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again")
    if gpu["vram_gb"] < 7.5:
        warn("less than 8 GB of VRAM: it runs, but few experts fit on the GPU and it will be slow")
    ram = S.ram_gb()
    need = min(d["ram_gb"] for d in MODELS.values())
    ok(f"RAM: {ram:.0f} GB")
    cpu, avx2, avx512 = S.cpu_info()
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'}), {physical_cores()} cores")
    if not avx2:
        fail("this CPU has no AVX2; the engine's expert kernels need it")
    if a.check:
        say()
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 6 else "does not fit"
            say(f"  {m:8s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        say("\nThis PC can run it. Run it again without --check to install.")
        return 0
    if ram < need - 6:
        fail(f"RAM: {ram:.0f} GB - the smallest size needs about {need} GB",
             "the engine keeps all of the model's experts in RAM (~20 GB) and the GPU a copy of the most-used ones")

    # ---- 2. the questions
    step(2, "your choices")
    names = list(MODELS)
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        say(f"  {i}) {m:8s} {d['about']}; download {d['bytes'] / 1e9:.0f} GB, uses ~{d['experts_gb']:.0f} GB of RAM{fit}")
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], "1", a.yes)) - 1]
    if ram < MODELS[model]["ram_gb"] - 2 and ask(f"  {model} needs about {MODELS[model]['ram_gb']} GB of RAM and this "
                                                 f"PC has {ram:.0f} GB. Install it anyway?", ["y", "n"], "n", a.yes) != "y":
        fail(f"{model} needs about {MODELS[model]['ram_gb']} GB of RAM")
    ok(f"size: {model}")
    rec_ctx = 32768 if gpu["vram_gb"] < 14 else 65536 if gpu["vram_gb"] < 20 else 131072
    if a.context:
        ctx = a.context
        if not 1024 <= ctx <= 262144:
            fail(f"--context {ctx}: from 1024 to 262144 tokens (the model's trained length)")
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it (about 22 KB per token), so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            say(f"  {i}) {c // 1024}K tokens ({c * KV_BYTES_PER_TOKEN / 2**30:.1f} GB of VRAM)"
                + ("   (recommended for your GPU)" if c == rec_ctx else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    ok(f"context: {ctx} tokens")
    data, _ = S.data_folder(a.data_dir)
    models_dir = data / "models"
    gguf = Path(a.gguf).expanduser().resolve() if a.gguf else None
    if gguf is not None:
        if not gguf.is_file():
            fail(f"not found: {gguf}")
        check_gguf(gguf)
    else:
        target = models_dir / model / MODELS[model]["file"]
        free_need = 0 if target.exists() and S.done(target) else MODELS[model]["bytes"] / 1e9 + 2
        if S.free_gb(models_dir) < free_need:
            fail(f"not enough free disk space in {models_dir}: need ~{free_need:.0f} GB", "use --data-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES,
                  "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    # ---- 4. the engine
    step(4, "the engine")
    eng = build_engine(gpu, a.yes)
    meta = json.loads((eng / "BUILD.json").read_text())

    # ---- 5. the model file
    step(5, f"downloading Qwen3.6-35B-A3B {model}")
    if gguf is None:
        gguf = get_model(model, models_dir)
    else:
        ok(f"using {gguf}")

    # ---- 6. the tokenizer
    step(6, "the tokenizer")
    pack = data / "packs" / tag(model if not a.gguf else gguf.stem.lower())
    if not (pack / "tokenizer" / "vocab.json").exists() or not (pack / "tokenizer" / "chat_template.jinja").exists():
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)])
    ok(f"tokenizer: {pack / 'tokenizer'}")

    # ---- 7. the start script
    step(7, "writing the start script")
    name = tag(model) if not a.gguf else tag(gguf.stem.lower())
    cfg_path = ROOT / f"strata-{name}.json"
    old = json.loads(cfg_path.read_text(encoding="utf-8-sig")) if cfg_path.exists() else {}
    port = a.port or old.get("port") or DEFAULT_PORT
    threads = a.threads or physical_cores()
    args = ["-m", str(gguf), "--expert-profile", str(PROFILE),
            "--profile-save", str(data / "expert-profile-qwen36.local.bin"),
            "--max-context", str(ctx), "--threads", str(threads)]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"qwen3.6-35b-a3b-{model.lower()}" if not a.gguf else gguf.stem.lower(),
           "log": str(ROOT / f"strata-{name}.log"), "lib_dirs": meta.get("cuda_dirs") or [], "port": port,
           # Qwen3.6's template thinks or does not: the web app's low / medium become thinking budgets (high: none)
           "reasoning_budget_by_effort": old.get("reasoning_budget_by_effort", EFFORT_BUDGETS)}
    if gpu["count"] > 1:
        cfg["gpu"] = gpu["index"]                       # the server tells the engine this card (CUDA_VISIBLE_DEVICES)
    for k in ("host", "api_key"):
        if getattr(a, k) or old.get(k):
            cfg[k] = getattr(a, k) or old.get(k)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    script = S.write_run_script(name, cfg_path, port)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key unless you set one; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    py = r".venv\Scripts\python" if WIN else ".venv/bin/python"
    say(f"  Chat in the terminal: {py} chat.py --port {port}")
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
