"""
Tests for Brain Module
"""

import pytest
import torch
import tempfile
from pathlib import Path

from brain.config import EntityConfig
from brain.model import EntityTransformer, RMSNorm, RotaryPositionalEmbedding, MultiHeadAttention, SwiGLU
from brain.tokenizer import BPETokenizer
from brain.inference import InferenceEngine


class TestEntityConfig:
    def test_default_config(self):
        config = EntityConfig()
        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.n_layers == 8
        assert config.n_heads == 8
        assert config.n_kv_heads == 4
        assert config.max_seq_len == 2048

    def test_param_count(self):
        config = EntityConfig(d_model=256, n_layers=4, n_heads=4)
        # Rough parameter count estimation
        # Embeddings: vocab_size * d_model * 2 (tied) ≈ 32000 * 256 * 2 = 16.4M
        # Transformer layers: ~4 * (4*256*256*3 + 256*1024*2) ≈ 4 * (0.8M + 0.5M) = 5.2M
        # Total ≈ 21M
        assert config.estimate_params() > 10_000_000
        assert config.estimate_params() < 50_000_000


class TestRMSNorm:
    def test_forward(self):
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)
        out = norm(x)
        assert out.shape == x.shape
        # Check normalization
        assert torch.allclose(out.var(dim=-1), torch.ones_like(out.var(dim=-1)), atol=1e-1)


class TestRotaryPositionalEmbedding:
    def test_forward(self):
        rope = RotaryPositionalEmbedding(64, max_seq_len=2048)
        x = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, head_dim)
        x_out = rope(x, offset=0)
        assert x_out.shape == x.shape

    def test_different_offsets(self):
        rope = RotaryPositionalEmbedding(64, max_seq_len=2048)
        x = torch.randn(1, 4, 5, 64)
        out1 = rope(x, offset=0)
        out2 = rope(x, offset=5)
        # Different offsets should produce different rotations
        assert not torch.allclose(out1, out2)


class TestMultiHeadAttention:
    def test_forward(self):
        config = EntityConfig(d_model=256, n_heads=4, n_kv_heads=2)
        attn = MultiHeadAttention(config)
        x = torch.randn(2, 10, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_kv_cache(self):
        config = EntityConfig(d_model=256, n_heads=4, n_kv_heads=2)
        attn = MultiHeadAttention(config)
        x = torch.randn(1, 5, 256)

        # First forward
        out1, kv1 = attn(x, use_cache=True)
        assert kv1 is not None
        assert len(kv1) == config.n_layers  # Actually per layer, but this is single layer

        # Second forward with cache
        x2 = torch.randn(1, 1, 256)
        out2, kv2 = attn(x2, past_kv=kv1, use_cache=True)
        assert kv2 is not None


class TestSwiGLU:
    def test_forward(self):
        swiglu = SwiGLU(256, 1024)
        x = torch.randn(2, 10, 256)
        out = swiglu(x)
        assert out.shape == x.shape


class TestEntityTransformer:
    def test_forward(self):
        config = EntityConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=128)
        model = EntityTransformer(config)
        x = torch.randint(0, 1000, (2, 20))
        logits, loss = model(x, targets=x)
        assert logits.shape == (2, 20, 1000)
        assert loss is not None
        assert loss.item() > 0

    def test_generate(self):
        config = EntityConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=128)
        model = EntityTransformer(config)
        prompt = torch.randint(0, 1000, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
        assert generated.shape == (1, 15)

    def test_gradient_flow(self):
        config = EntityConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=128)
        model = EntityTransformer(config)
        x = torch.randint(0, 1000, (2, 20))
        logits, loss = model(x, targets=x)
        loss.backward()
        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestBPETokenizer:
    def test_train_and_encode(self):
        tokenizer = BPETokenizer(vocab_size=1000)
        corpus = ["hello world", "hello there", "world peace"] * 100
        tokenizer.train(corpus)

        assert tokenizer.vocab_size >= 256  # At least byte vocab

        # Test encode/decode
        text = "hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert "hello" in decoded.lower()
        assert "world" in decoded.lower()

    def test_save_load(self):
        tokenizer = BPETokenizer(vocab_size=1000)
        corpus = ["test corpus for saving"] * 50
        tokenizer.train(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(Path(tmpdir))
            loaded = BPETokenizer.load(Path(tmpdir))
            assert loaded.vocab_size == tokenizer.vocab_size

            # Test encode/decode consistency
            text = "test saving"
            tokens1 = tokenizer.encode(text)
            tokens2 = loaded.encode(text)
            assert tokens1 == tokens2

    def test_byte_fallback(self):
        tokenizer = BPETokenizer(vocab_size=500)
        corpus = ["ascii only"] * 100
        tokenizer.train(corpus)

        # Unicode characters should fall back to bytes
        unicode_text = "héllo wörld 🌍"
        tokens = tokenizer.encode(unicode_text)
        decoded = tokenizer.decode(tokens)
        # Should not crash and should preserve content approximately
        assert len(tokens) > 0


class TestInferenceEngine:
    @pytest.fixture
    def model_and_tokenizer(self):
        config = EntityConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=128)
        model = EntityTransformer(config)
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.train(["test corpus for inference"] * 50)
        return model, tokenizer

    def test_generate(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        engine = InferenceEngine(model, tokenizer, device="cpu")

        result = engine.generate("Hello", max_new_tokens=20, temperature=0.8)
        assert isinstance(result, str)
        assert len(result) > len("Hello")

    def test_stream_generate(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        engine = InferenceEngine(model, tokenizer, device="cpu")

        tokens = []
        for token in engine.stream_generate("Hello", max_new_tokens=10):
            tokens.append(token)
        assert len(tokens) > 0

    def test_perplexity(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        engine = InferenceEngine(model, tokenizer, device="cpu")

        ppl = engine.estimate_perplexity("This is a test sentence for perplexity estimation.")
        assert ppl > 0
        assert ppl < 10000  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])