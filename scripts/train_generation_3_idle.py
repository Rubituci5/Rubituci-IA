#!/usr/bin/env python3
"""Train generation 3 in resumable, production-aware CPU blocks.

The script exits without training when the host is busy, memory/disk is low, or
the public web access log changes while a block is running. It never promotes a
checkpoint to production.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.config import EntityConfig
from brain.model import EntityTransformer
from brain.tokenizer import EntityTokenizer

GEN_DIR = ROOT / "snapshots" / "generation_000003"
CHECKPOINT = GEN_DIR / "idle_checkpoint.pt"
REPORT = GEN_DIR / "idle_training_report.json"
CACHE = GEN_DIR / "training_tokens.pt"
ACCESS_LOG = Path(os.environ.get("RUBITUCI_ACCESS_LOG", "/var/log/nginx/access.log"))
SEQUENCE_LENGTH = 128


def load_portuguese_corpus() -> list[str]:
    texts: list[str] = []
    for path in sorted((ROOT / "data").glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    for path in sorted((ROOT / "data").glob("**/*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        texts.extend(content[index:index + 4000] for index in range(0, len(content), 4000))
    return texts


def host_ready() -> tuple[bool, str]:
    load_one = os.getloadavg()[0]
    memory = Path("/proc/meminfo")
    if memory.exists():
        values = {line.split(":", 1)[0]: int(line.split()[1]) for line in memory.read_text().splitlines() if ":" in line}
        if values.get("MemAvailable", 0) < 6 * 1024 * 1024:
            return False, "menos de 6 GiB de RAM disponíveis"
    if load_one > 1.25:
        return False, f"carga do servidor alta ({load_one:.2f})"
    if shutil.disk_usage(GEN_DIR).free < 3 * 1024**3:
        return False, "menos de 3 GiB livres em disco"
    return True, "servidor ocioso"


def make_stream(tokenizer: EntityTokenizer) -> torch.Tensor:
    if CACHE.exists():
        stream = torch.load(CACHE, map_location="cpu", weights_only=True)
        if isinstance(stream, torch.Tensor) and stream.numel() > SEQUENCE_LENGTH:
            print(f"Cache carregado: {stream.numel():,} tokens", flush=True)
            return stream
    ids: list[int] = []
    for text in load_portuguese_corpus():
        ids.extend(tokenizer.encode(text, add_special_tokens=True))
    if len(ids) <= SEQUENCE_LENGTH:
        raise RuntimeError("Corpus insuficiente")
    stream = torch.tensor(ids, dtype=torch.long)
    temporary = CACHE.with_suffix(".tmp")
    torch.save(stream, temporary)
    temporary.replace(CACHE)
    print(f"Cache criado: {stream.numel():,} tokens", flush=True)
    return stream


def save_checkpoint(model: EntityTransformer, optimizer: torch.optim.Optimizer, config: EntityConfig, state: dict) -> None:
    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": config.to_dict(), **state}, temporary)
    temporary.replace(CHECKPOINT)
    REPORT.write_text(json.dumps({key: value for key, value in state.items() if isinstance(value, (str, int, float, bool, type(None)))}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    args = parser.parse_args()
    ready, reason = host_ready()
    if not ready and not args.initialize_only and not args.prepare_cache:
        print(f"Bloco adiado: {reason}", flush=True)
        return
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    random.seed(42)
    torch.manual_seed(42)
    tokenizer = EntityTokenizer.from_pretrained(GEN_DIR / "tokenizer")
    config = EntityConfig.load(GEN_DIR / "config.json")
    if args.prepare_cache:
        make_stream(tokenizer)
        return
    model = EntityTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    state = {"generation": 3, "step": 0, "tokens_seen": 0, "last_loss": None, "promotion_allowed": False}
    if CHECKPOINT.exists():
        previous = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(previous["model_state_dict"])
        if previous.get("optimizer_state_dict"):
            optimizer.load_state_dict(previous["optimizer_state_dict"])
        state.update({key: previous.get(key, value) for key, value in state.items()})
    if args.initialize_only:
        save_checkpoint(model, optimizer, config, state)
        print(f"Checkpoint inicial criado com {sum(parameter.numel() for parameter in model.parameters()):,} parâmetros", flush=True)
        return
    stream = make_stream(tokenizer)
    access_mtime = ACCESS_LOG.stat().st_mtime if ACCESS_LOG.exists() else 0
    model.train()
    for _ in range(max(1, args.steps)):
        if ACCESS_LOG.exists() and ACCESS_LOG.stat().st_mtime > access_mtime:
            print("Bloco interrompido: novo acesso ao site", flush=True)
            break
        start = random.randint(0, len(stream) - SEQUENCE_LENGTH - 1)
        batch = stream[start:start + SEQUENCE_LENGTH + 1].unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        state["step"] += 1
        state["tokens_seen"] += SEQUENCE_LENGTH
        state["last_loss"] = float(loss.item())
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"step={state['step']} tokens={state['tokens_seen']} loss={state['last_loss']:.4f}", flush=True)
    save_checkpoint(model, optimizer, config, state)


if __name__ == "__main__":
    main()
