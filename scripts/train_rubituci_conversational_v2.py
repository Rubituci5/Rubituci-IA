#!/usr/bin/env python3

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


GEN_DIR = ROOT / "snapshots" / "generation_000001"
BACKUP_DIR = ROOT / "backups" / "stable"

BASE_MODEL = GEN_DIR / "rubituci_conversational_v2" / "model_weights_rubituci_conversational_v2.pt"
TOKENIZER_PATH = GEN_DIR / "tokenizer"
CONFIG_PATH = GEN_DIR / "config.json"

DATASET_PATH = (
    ROOT
    / "data"
    / "oasst1_ptbr"
    / "processed"
    / "rubituci_conversational_v2.jsonl"
)
LITERACY_DATASET_PATH = ROOT / "data" / "literacy_ptbr_v1" / "literacy_ptbr_v1.jsonl"

OUT_DIR = GEN_DIR / "rubituci_literacy_v1"
FINAL_PATH = OUT_DIR / "model_weights_rubituci_literacy_v1.pt"

IGNORE_INDEX = -100


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset():
    texts = []

    # Oversample the small reviewed literacy set so it is not drowned out by
    # general chat. Both sources remain human-readable JSONL for open review.
    for path, repeats in ((DATASET_PATH, 1), (LITERACY_DATASET_PATH, 4)):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            source_texts = []
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                if text:
                    source_texts.append(text)
            texts.extend(source_texts * repeats)

    return texts


def encode_example(tokenizer, text, max_seq_len):
    marker = "\nEntity:"

    if marker not in text:
        raise ValueError("Example without Entity marker")

    user_part, entity_part = text.split(marker, 1)

    prompt_text = user_part + marker
    full_text = prompt_text + entity_part

    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=True,
    )

    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=True,
    )

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    # labels iguais aos tokens, mas ignoramos tudo que pertence ao prompt
    labels = full_ids.copy()

    prompt_len = min(
        len(prompt_ids),
        len(labels),
    )

    for i in range(prompt_len):
        labels[i] = IGNORE_INDEX

    return full_ids, labels


def make_batch(tokenizer, texts, batch_size, device, max_seq_len):
    selected = random.choices(
        texts,
        k=batch_size,
    )

    encoded = [
        encode_example(
            tokenizer,
            text,
            max_seq_len,
        )
        for text in selected
    ]

    max_len = max(
        len(ids)
        for ids, _ in encoded
    )

    pad = tokenizer.pad_token_id

    input_batch = torch.full(
        (batch_size, max_len),
        pad,
        dtype=torch.long,
    )

    label_batch = torch.full(
        (batch_size, max_len),
        IGNORE_INDEX,
        dtype=torch.long,
    )

    for i, (ids, labels) in enumerate(encoded):
        input_batch[i, :len(ids)] = torch.tensor(
            ids,
            dtype=torch.long,
        )

        label_batch[i, :len(labels)] = torch.tensor(
            labels,
            dtype=torch.long,
        )

    inputs = input_batch[:, :-1].to(device)
    targets = label_batch[:, 1:].to(device)

    return inputs, targets


def save_checkpoint(model, config, step, loss_value):
    path = OUT_DIR / f"checkpoint_step_{step:04d}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "generation": 1,
            "training_type": "rubituci_conversational_v2",
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("RUBITUCI - PORTUGUESE LITERACY V1 TRAINING")
    print("=" * 60)

    device = choose_device()
    print("Device:", device)

    tokenizer = EntityTokenizer.from_pretrained(
        TOKENIZER_PATH
    )

    print("Tokenizer vocab:", tokenizer.vocab_size)

    texts = load_dataset()

    print("Training samples:", len(texts))
    print("Base model:", BASE_MODEL)

    config_data = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    config = EntityConfig.from_dict(
        config_data
    )

    model = EntityTransformer(
        config
    ).to(device)

    checkpoint = torch.load(
        BASE_MODEL,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6,
        weight_decay=0.01,
    )

    steps = 120
    batch_size = 2

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
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
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

        if step % 20 == 0:
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
            "training_type": "rubituci_conversational_v2",
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
