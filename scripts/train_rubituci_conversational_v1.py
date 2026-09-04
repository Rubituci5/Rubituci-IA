#!/usr/bin/env python3

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from brain.config import EntityConfig
from brain.model import EntityTransformer
from brain.tokenizer import EntityTokenizer


ROOT = Path(__file__).resolve().parent.parent

GEN_DIR = ROOT / "snapshots" / "generation_000001"
BACKUP_DIR = ROOT / "backups" / "stable"

BASE_MODEL = BACKUP_DIR / "gen1_stable_conversational.pt"
TOKENIZER_PATH = GEN_DIR / "tokenizer"
CONFIG_PATH = GEN_DIR / "config.json"

DATASET_PATH = (
    ROOT
    / "data"
    / "oasst1_ptbr"
    / "processed"
    / "rubituci_conversational_v1.jsonl"
)

OUT_DIR = GEN_DIR / "rubituci_conversational_v1"
FINAL_PATH = OUT_DIR / "model_weights_rubituci_conversational_v1.pt"


IDENTITY_EXAMPLES = [
    "User: Qual é o seu nome?\nRubituci: Meu nome é Rubituci.",
    "User: Quem é você?\nRubituci: Eu sou Rubituci, uma inteligência digital em desenvolvimento.",
    "User: Você é humana?\nRubituci: Não. Sou uma inteligência digital.",
    "User: O que você pode fazer?\nRubituci: Posso conversar, aprender com interações e utilizar memória persistente.",
    "User: Você se lembra das coisas?\nRubituci: Posso utilizar memórias armazenadas pelo meu sistema.",
    "User: Olá\nRubituci: Olá. Como posso ajudar?",
]


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset():
    rows = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            text = item.get("text", "").strip()

            if text:
                rows.append(text)

    # Reforço leve de identidade
    rows.extend(IDENTITY_EXAMPLES * 8)

    return rows


def make_batch(tokenizer, texts, batch_size, device, max_seq_len):
    selected = random.choices(texts, k=batch_size)

    encoded = []

    for text in selected:
        ids = tokenizer.encode(text, add_special_tokens=True)

        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        encoded.append(ids)

    max_len = max(len(x) for x in encoded)
    pad = tokenizer.pad_token_id

    batch = torch.full(
        (batch_size, max_len),
        pad,
        dtype=torch.long,
    )

    for i, ids in enumerate(encoded):
        batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)

    return inputs, targets


def save_checkpoint(model, config, step, loss_value):
    path = OUT_DIR / f"checkpoint_step_{step:04d}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "generation": 1,
            "training_type": "rubituci_conversational_v1",
            "step": step,
            "loss": loss_value,
            "base_model": str(BASE_MODEL),
            "dataset": str(DATASET_PATH),
        },
        path,
    )

    print("Checkpoint:", path)


def main():
    torch.manual_seed(42)
    random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUBITUCI - CONVERSATIONAL V1 TRAINING")
    print("=" * 60)

    device = choose_device()
    print("Device:", device)

    tokenizer = EntityTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer vocab:", tokenizer.vocab_size)

    texts = load_dataset()

    print("Conversational samples:", len(texts))
    print("Base model:", BASE_MODEL)

    config_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = EntityConfig.from_dict(config_data)

    model = EntityTransformer(config).to(device)

    checkpoint = torch.load(
        BASE_MODEL,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
    )

    steps = 150
    batch_size = 2
    pad_id = tokenizer.pad_token_id

    model.train()

    first_loss = None
    last_loss = None

    for step in range(1, steps + 1):
        inputs, targets = make_batch(
            tokenizer,
            texts,
            batch_size,
            device,
            config.max_seq_len,
        )

        optimizer.zero_grad()

        outputs = model(inputs)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        loss_value = loss.item()

        if first_loss is None:
            first_loss = loss_value

        last_loss = loss_value

        if step == 1 or step % 10 == 0:
            print(
                f"Step {step:04d}/{steps} | "
                f"Loss: {loss_value:.4f}"
            )

        if step % 25 == 0:
            save_checkpoint(
                model,
                config,
                step,
                loss_value,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "generation": 1,
            "training_type": "rubituci_conversational_v1",
            "step": steps,
            "base_model": str(BASE_MODEL),
            "dataset": str(DATASET_PATH),
            "first_loss": first_loss,
            "final_loss": last_loss,
        },
        FINAL_PATH,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Saved:", FINAL_PATH)
    print("Initial loss:", first_loss)
    print("Final loss:", last_loss)


if __name__ == "__main__":
    main()
