"""
Entity Transformer Model

A from-scratch implementation of a decoder-only Transformer language model.
No pre-trained weights or external model dependencies.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import EntityConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positional embeddings."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin values for the absolute token positions."""
        required_len = position_offset + seq_len

        if required_len > self.max_seq_len:
            self._build_cache(required_len)
            self.max_seq_len = required_len

        return (
            self.cos_cached[:, :, position_offset:required_len, :].to(x.device),
            self.sin_cached[:, :, position_offset:required_len, :].to(x.device),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key."""
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    q_rot = q[:, :, :, :cos.shape[-1]]
    k_rot = k[:, :, :, :cos.shape[-1]]
    q_pass = q[:, :, :, cos.shape[-1]:]
    k_pass = k[:, :, :, cos.shape[-1]:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat([q_embed, q_pass], dim=-1)
    k = torch.cat([k_embed, k_pass], dim=-1)
    return q, k


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE."""

    def __init__(self, config: EntityConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_seq_len
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Absolute position offset when decoding with KV cache.
        # Without this, every newly generated token would incorrectly
        # receive RoPE position 0.
        past_len = (
            past_key_value[0].size(2)
            if past_key_value is not None
            else 0
        )

        # Apply RoPE using absolute positions.
        cos, sin = self.rotary_emb(
            x,
            seq_len,
            position_offset=past_len,
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle past key/value for inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Cache current key/value if needed
        present_key_value = (k, v) if use_cache else None

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        # When using KV cache, query positions are offset by the number
        # of tokens already present in the cache.
        total_kv_len = k.size(2)
        past_len = total_kv_len - seq_len

        query_positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=x.device,
        ).unsqueeze(1)

        key_positions = torch.arange(
            total_kv_len,
            device=x.device,
        ).unsqueeze(0)

        causal_mask = key_positions > query_positions

        attn_weights.masked_fill_(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        # Additional attention mask (padding)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)
        return output, present_key_value


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation."""

    def __init__(self, config: EntityConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer Block (Pre-Norm)."""

    def __init__(self, config: EntityConfig):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with residual
        residual = x
        x = self.attention_norm(x)
        x, present_key_value = self.attention(
            x, attention_mask, past_key_value, use_cache
        )
        x = residual + x

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, present_key_value


class EntityTransformer(nn.Module):
    """
    Entity Transformer Language Model.

    A decoder-only Transformer trained from scratch.
    No pre-trained weights or external dependencies.
    """

    def __init__(self, config: EntityConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.d_model)

        # Output head (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        self._print_parameter_count()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_parameter_count(self) -> None:
        """Log parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EntityTransformer initialized:")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
            past_key_values: List of cached key/values for each layer
            use_cache: Whether to return key/values for next step
            return_dict: Whether to return dict or tuple

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_key_values: List of (key, value) for each layer if use_cache
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Prepare attention mask
        if attention_mask is not None:
            # Convert to additive mask: 0 -> 0, 1 -> -inf
            attention_mask = attention_mask[:, None, None, :].float()
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Process through layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(
                x, attention_mask, past_kv, use_cache
            )
            if use_cache:
                present_key_values.append(present_kv)

        # Final norm and output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Causal language-model loss.
        # Token at position t predicts token at position t+1.
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        if targets is not None:
            return logits, loss

        if return_dict:
            return {
                "logits": logits,
                "past_key_values": present_key_values,
            }

        return logits, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            do_sample: Whether to sample or greedy decode

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.size(0)
        past_key_values = None

        # Track generated tokens for repetition penalty
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass with cache
            outputs = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample and temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            input_ids = next_token

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


def create_model(config: Optional[EntityConfig] = None, **kwargs) -> EntityTransformer:
    """Factory function to create model."""
    if config is None:
        config = EntityConfig(**kwargs)
    return EntityTransformer(config)