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

MODEL_PATH = GEN_DIR / "model_weights_trained.pt"
TOKENIZER_PATH = GEN_DIR / "tokenizer"
CONFIG_PATH = GEN_DIR / "config.json"

DATASET_PATH = ROOT / "data" / "foundation_v1" / "processed" / "foundation_v1.jsonl"
OUTPUT_PATH = GEN_DIR / "model_weights_foundation_v1.pt"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset():
    texts = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            text = data.get("text", "").strip()

            if text:
                texts.append(text)

    return texts


def make_batch(tokenizer, texts, batch_size, device, max_seq_len):
    selected = random.choices(texts, k=batch_size)
    encoded = []

    for text in selected:
        ids = tokenizer.encode(text, add_special_tokens=True)

        if len(ids) > max_seq_len:
            start = random.randint(0, len(ids) - max_seq_len)
            ids = ids[start:start + max_seq_len]

        encoded.append(ids)

    max_len = max(len(x) for x in encoded)
    pad = tokenizer.pad_token_id

    batch = torch.full(
        (batch_size, max_len),
        pad,
        dtype=torch.long,
    )

    for i, tokens in enumerate(encoded):
        batch[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)

    return inputs, targets


def main():
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("ENTITY GEN 000001 - FOUNDATION V1 TRAINING")
    print("=" * 60)

    device = choose_device()
    print("Device:", device)

    tokenizer = EntityTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer vocab:", tokenizer.vocab_size)

    texts = load_dataset()
    print("Foundation records:", len(texts))

    if not texts:
        raise RuntimeError("Foundation dataset is empty")

    config_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = EntityConfig.from_dict(config_data)

    model = EntityTransformer(config).to(device)

    print("Loading base checkpoint:", MODEL_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
    )

    steps = 1000
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

        if step == 1 or step % 25 == 0:
            print(
                f"Step {step:04d}/{steps} | "
                f"Loss: {loss_value:.4f}"
            )

    trained_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation": 1,
        "step": steps,
        "training_type": "foundation_v1",
        "base_checkpoint": str(MODEL_PATH),
        "dataset": str(DATASET_PATH),
        "foundation_records": len(texts),
        "learning_rate": 5e-5,
        "first_loss": first_loss,
        "final_loss": last_loss,
    }

    torch.save(
        trained_checkpoint,
        OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("FOUNDATION TRAINING COMPLETE")
    print("=" * 60)
    print("Saved:", OUTPUT_PATH)
    print("Initial loss:", first_loss)
    print("Final loss:", last_loss)


if __name__ == "__main__":
    main()
