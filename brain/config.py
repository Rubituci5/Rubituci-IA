"""
Entity Model Configuration

Configuration class for the Entity Transformer model.
All hyperparameters are defined here for easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class EntityConfig:
    """Configuration for the Entity Transformer model."""

    # Model architecture
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1

    # Tokenizer
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    grad_accum_steps: int = 4
    batch_size: int = 4
    eval_every: int = 500
    save_every: int = 1000
    seed: int = 42

    # Inference
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256

    # Generation info
    generation: int = 1
    model_name: str = "entity-gen-000001"
    created_at: Optional[str] = None
    git_commit: Optional[str] = None
    training_data_hash: Optional[str] = None

    # Metadata
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def num_parameters(self) -> int:
        """Estimate trainable parameters for the current tied-head RoPE model."""
        # Token embeddings are shared with the output head; RoPE has no weights.
        embed_params = self.vocab_size * self.d_model
        # Transformer layers
        layer_params = self.n_layers * (
            # Attention: Q, K, V, O projections
            4 * self.d_model * self.d_model +
            # SwiGLU FFN: gate, up, and down projections
            3 * self.d_model * self.d_ff +
            # Two RMSNorms per layer
            2 * self.d_model
        )
        final_params = self.d_model
        return embed_params + layer_params + final_params

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "batch_size": self.batch_size,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "generation": self.generation,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "training_data_hash": self.training_data_hash,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EntityConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        params = self.num_parameters
        if params >= 1_000_000:
            param_str = f"{params / 1_000_000:.1f}M"
        else:
            param_str = f"{params / 1_000:.1f}K"
        return f"EntityConfig({param_str} params, {self.n_layers}L/{self.n_heads}H/{self.d_model}D)"
