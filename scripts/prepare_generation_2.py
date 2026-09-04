#!/usr/bin/env python3
"""Prepare Rubituci generation 2 tokenizer and architecture manifest.

This intentionally does not promote an untrained model. A tokenizer change
requires training generation 2 from scratch and passing evaluation first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.config import EntityConfig
from brain.tokenizer import EntityTokenizer


OUTPUT = ROOT / "snapshots" / "generation_000002"
TOKENIZER_DIR = OUTPUT / "tokenizer"


def load_portuguese_corpus() -> list[str]:
    texts: list[str] = []
    for path in sorted((ROOT / "data").glob("**/*.jsonl")):
        with path.open(encoding="utf-8", errors="ignore") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") or row.get("content")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    for path in sorted((ROOT / "data").glob("**/*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            # Smaller passages keep word frequencies useful and memory bounded.
            texts.extend(content[index:index + 4000] for index in range(0, len(content), 4000))
    return texts


def main() -> None:
    corpus = load_portuguese_corpus()
    if not corpus:
        raise RuntimeError("Nenhum corpus português foi encontrado")
    tokenizer = EntityTokenizer.train_new(corpus, vocab_size=4096, min_frequency=2)
    tokenizer.save(TOKENIZER_DIR)
    config = EntityConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1536,
        max_seq_len=1024,
        batch_size=2,
        grad_accum_steps=8,
        generation=2,
        model_name="rubituci-gen-000002",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "generation": 2,
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
