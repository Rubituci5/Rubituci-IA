#!/usr/bin/env python3
"""Short supervised dialogue alignment pass for generation 2."""

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

GEN = ROOT / "snapshots" / "generation_000002"
MODEL_PATH = GEN / "model_weights_rubituci_gen2.pt"
DATA = ROOT / "data" / "oasst1_ptbr" / "processed" / "rubituci_conversational_v2.jsonl"
STEPS = 320


def device():
    return torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")


def examples(tokenizer):
    rows = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        text = json.loads(line).get("text", "")
        if "\nEntity:" not in text:
            continue
        prompt, answer = text.split("\nEntity:", 1)
        prefix = tokenizer.encode(prompt + "\nEntity:", add_special_tokens=True)
        full = tokenizer.encode(prompt + "\nEntity:" + answer, add_special_tokens=True)[:512]
        labels = full.copy()
        labels[:min(len(prefix), len(labels))] = [-100] * min(len(prefix), len(labels))
        if any(value != -100 for value in labels[1:]):
            rows.append((full, labels))
    return rows


def main():
    target = device()
    tokenizer = EntityTokenizer.from_pretrained(GEN / "tokenizer")
    config = EntityConfig.load(GEN / "config.json")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = EntityTransformer(config).to(target)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = examples(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    model.train()
    for step in range(1, STEPS + 1):
        chosen = random.choices(rows, k=4)
        length = max(len(item[0]) for item in chosen)
        inputs = torch.full((4, length), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((4, length), -100, dtype=torch.long)
        for index, (ids, target_ids) in enumerate(chosen):
            inputs[index, :len(ids)] = torch.tensor(ids)
            labels[index, :len(target_ids)] = torch.tensor(target_ids)
        inputs, labels = inputs.to(target), labels.to(target)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs[:, :-1])["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 40 == 0:
            print(f"step={step}/{STEPS} loss={loss.item():.4f}", flush=True)
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["dialogue_finetune_steps"] = STEPS
    checkpoint["dialogue_finetune_loss"] = float(loss.item())
    checkpoint["training_type"] = "causal_portuguese_plus_dialogue"
    torch.save(checkpoint, MODEL_PATH)


if __name__ == "__main__":
    main()
