#!/usr/bin/env python3
"""Train Rubituci generation 2 from scratch on at least 100k tokens."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.config import EntityConfig
from brain.model import EntityTransformer
from brain.tokenizer import EntityTokenizer
from scripts.prepare_generation_2 import load_portuguese_corpus


GEN_DIR = ROOT / "snapshots" / "generation_000002"
MODEL_PATH = GEN_DIR / "model_weights_rubituci_gen2.pt"
REPORT_PATH = GEN_DIR / "training_report.json"
MINIMUM_TOKENS = 10_000_000
SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
STEPS = 11264


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_stream(tokenizer: EntityTokenizer, texts: list[str]) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_special_tokens=True))
    if len(ids) < SEQUENCE_LENGTH + 1:
        raise RuntimeError("Corpus insuficiente")
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(stream: torch.Tensor, device: torch.device) -> torch.Tensor:
    starts = [random.randint(0, len(stream) - SEQUENCE_LENGTH - 1) for _ in range(BATCH_SIZE)]
    return torch.stack([stream[start:start + SEQUENCE_LENGTH + 1] for start in starts]).to(device)


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = choose_device()
    tokenizer = EntityTokenizer.from_pretrained(GEN_DIR / "tokenizer")
    config = EntityConfig.load(GEN_DIR / "config.json")
    corpus = load_portuguese_corpus()
    random.shuffle(corpus)
    split = max(1, int(len(corpus) * 0.95))
    train_stream = make_stream(tokenizer, corpus[:split])
    eval_stream = make_stream(tokenizer, corpus[split:])
    model = EntityTransformer(config).to(device)
    start_step = 0
    tokens_seen = 0
    first_loss = None
    best_eval_loss = float("inf")
    if MODEL_PATH.exists():
        previous = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(previous["model_state_dict"])
        start_step = int(previous.get("step", 0))
        tokens_seen = int(previous.get("tokens_seen", 0))
        first_loss = previous.get("first_loss")
        best_eval_loss = float(previous.get("best_eval_loss", best_eval_loss))
        print(f"Retomando step={start_step}, tokens={tokens_seen}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    last_loss = None
    model.train()
    for step in range(start_step + 1, STEPS + 1):
        batch = sample_batch(train_stream, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch[:, :-1])
        logits = output["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += batch[:, 1:].numel()
        last_loss = float(loss.item())
        first_loss = last_loss if first_loss is None else first_loss
        if step == 1 or step % 32 == 0:
            model.eval()
            with torch.no_grad():
                validation = sample_batch(eval_stream, device)
                validation_logits = model(validation[:, :-1])["logits"]
                eval_loss = float(F.cross_entropy(validation_logits.reshape(-1, validation_logits.size(-1)), validation[:, 1:].reshape(-1)).item())
            best_eval_loss = min(best_eval_loss, eval_loss)
            model.train()
            print(f"step={step}/{STEPS} tokens={tokens_seen} train_loss={last_loss:.4f} eval_loss={eval_loss:.4f}", flush=True)
        if step % 256 == 0:
            torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict(), "generation": 2, "training_type": "causal_portuguese_10m_plus", "step": step, "tokens_seen": tokens_seen, "first_loss": first_loss, "final_loss": last_loss, "best_eval_loss": best_eval_loss}, MODEL_PATH)
    if tokens_seen < MINIMUM_TOKENS:
        raise RuntimeError(f"Treinamento incompleto: apenas {tokens_seen} tokens")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation": 2,
        "training_type": "causal_portuguese_10m_plus",
        "step": STEPS,
        "tokens_seen": tokens_seen,
        "first_loss": first_loss,
        "final_loss": last_loss,
        "best_eval_loss": best_eval_loss,
    }
    torch.save(checkpoint, MODEL_PATH)
    report = {key: checkpoint[key] for key in ("generation", "training_type", "step", "tokens_seen", "first_loss", "final_loss", "best_eval_loss")}
    report.update({"device": str(device), "training_samples": len(corpus), "vocabulary_size": tokenizer.vocab_size, "context_tokens": config.max_seq_len, "model_path": str(MODEL_PATH)})
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
