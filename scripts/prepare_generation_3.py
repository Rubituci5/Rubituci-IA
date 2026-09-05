#!/usr/bin/env python3
"""Prepare the tokenizer and 150M-parameter manifest for Rubituci generation 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.config import EntityConfig
from brain.tokenizer import EntityTokenizer
from scripts.prepare_generation_2 import load_portuguese_corpus

OUTPUT = ROOT / "snapshots" / "generation_000003"
TOKENIZER_DIR = OUTPUT / "tokenizer"


def main() -> None:
    corpus = load_portuguese_corpus()
    if not corpus:
        raise RuntimeError("Nenhum corpus português foi encontrado")
    tokenizer = EntityTokenizer.train_new(corpus, vocab_size=8_192, min_frequency=1)
    tokenizer.save(TOKENIZER_DIR)
    config = EntityConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_layers=15,
        n_heads=12,
        d_ff=3072,
        max_seq_len=2048,
        batch_size=1,
        grad_accum_steps=16,
        generation=3,
        model_name="rubituci-gen-000003-150m",
        extra={"promotion_allowed": False, "training_mode": "idle_cpu_blocks"},
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config.save(OUTPUT / "config.json")
    report = {
        "generation": 3,
        "training_samples": len(corpus),
        "vocabulary_size": tokenizer.vocab_size,
        "context_tokens": config.max_seq_len,
        "estimated_parameters": config.num_parameters,
        "status": "prepared_not_trained",
        "promotion_allowed": False,
    }
    (OUTPUT / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
