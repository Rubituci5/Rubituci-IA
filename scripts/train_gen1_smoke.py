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
MODEL_PATH = GEN_DIR / "model_weights.pt"
TOKENIZER_PATH = GEN_DIR / "tokenizer"
CONFIG_PATH = GEN_DIR / "config.json"
TRAINED_PATH = GEN_DIR / "model_weights_trained.pt"


TEXTS = [
    "User: Hello\nEntity: Hello. How can I help you?",
    "User: Hello\nEntity: Hello. I am Entity.",
    "User: Who are you?\nEntity: I am Entity, a developing digital intelligence.",
    "User: What is your name?\nEntity: My name is Entity.",
    "User: What can you do?\nEntity: I can communicate, learn from interactions, and use persistent memory.",
    "User: Do you remember things?\nEntity: Yes. I can use memories stored by my system.",
    "User: Are you human?\nEntity: No. I am a digital intelligence.",
    "User: What is memory?\nEntity: Memory helps me preserve information from previous experiences.",

    "User: Olá\nEntity: Olá. Como posso ajudar?",
    "User: Quem é você?\nEntity: Eu sou a Entity, uma inteligência digital em desenvolvimento.",
    "User: Qual é o seu nome?\nEntity: Meu nome é Entity.",
    "User: Você se lembra das coisas?\nEntity: Sim. Posso utilizar memórias armazenadas pelo meu sistema.",
    "User: Você é humana?\nEntity: Não. Sou uma inteligência digital.",
    "User: O que você pode fazer?\nEntity: Posso conversar, aprender com interações e utilizar memória persistente.",
] * 100


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_batch(tokenizer, texts, batch_size, device):
    selected = random.choices(texts, k=batch_size)

    encoded = [
        tokenizer.encode(text, add_special_tokens=True)
        for text in selected
    ]

    max_len = max(len(x) for x in encoded)

    pad = tokenizer.pad_token_id

    batch = torch.full(
        (batch_size, max_len),
        pad,
        dtype=torch.long,
    )

    for i, tokens in enumerate(encoded):
        batch[i, :len(tokens)] = torch.tensor(tokens)

    inputs = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)

    return inputs, targets


def main():
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 60)
    print("ENTITY GEN 000001 - SMOKE TRAINING")
    print("=" * 60)

    device = choose_device()
    print("Device:", device)

    tokenizer = EntityTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer vocab:", tokenizer.vocab_size)

    config_data = json.loads(CONFIG_PATH.read_text())
    config = EntityConfig.from_dict(config_data)

    model = EntityTransformer(config).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    steps = 500
    batch_size = 8
    pad_id = tokenizer.pad_token_id

    model.train()

    for step in range(1, steps + 1):
        inputs, targets = make_batch(
            tokenizer,
            TEXTS,
            batch_size,
            device,
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

        if step == 1 or step % 25 == 0:
            print(
                f"Step {step:04d}/{steps} | "
                f"Loss: {loss.item():.4f}"
            )

    trained_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation": 1,
        "step": steps,
        "training_type": "smoke_test",
    }

    torch.save(trained_checkpoint, TRAINED_PATH)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Saved:", TRAINED_PATH)


if __name__ == "__main__":
    main()
