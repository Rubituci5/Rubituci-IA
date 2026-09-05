"""
Entity Tokenizer - Byte Pair Encoding (BPE) Implementation

A from-scratch BPE tokenizer for the Entity project.
No external tokenizer dependencies (no tiktoken, no HuggingFace tokenizers).
"""

import json
import regex as re
from functools import lru_cache
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import torch


@dataclass
class TokenizerConfig:
    """Configuration for the Entity tokenizer."""
    vocab_size: int = 8192
    min_frequency: int = 2
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<mask>": 4,
    })
    # Pattern for pre-tokenization (GPT-2 style)
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    byte_fallback: bool = True

    def __post_init__(self):
        # Ensure special tokens are at the beginning
        max_special = max(self.special_tokens.values())
        assert max_special < self.vocab_size, "Special tokens must fit in vocab_size"


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer implemented from scratch.

    Features:
    - BPE merge rules learned from corpus
    - Byte-level fallback for unknown characters
    - Special tokens support
    - Fast encoding/decoding
    - Compatible with PyTorch
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.pattern = re.compile(self.config.pattern)
        self._initialized = False

        # Add special tokens first
        for token, idx in self.config.special_tokens.items():
            self.vocab[token] = idx
            self.id_to_token[idx] = token

        self._next_id = max(self.config.special_tokens.values()) + 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.config.special_tokens["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.config.special_tokens["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.config.special_tokens["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.config.special_tokens["<eos>"]

    @property
    def mask_token_id(self) -> int:
        return self.config.special_tokens.get("<mask>", 4)

    def train(self, texts: List[str], vocab_size: Optional[int] = None, verbose: bool = True) -> None:
        """
        Train BPE tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size (default: config.vocab_size)
            verbose: Print progress
        """
        target_vocab = vocab_size or self.config.vocab_size
        if verbose:
            print(f"Training BPE tokenizer on {len(texts)} texts...")
            print(f"Target vocab size: {target_vocab}")

        # Pre-tokenize all texts
        words = []
        for text in texts:
            words.extend(self._pre_tokenize(text))

        # Count word frequencies.
        # Internally BPE operates on space-separated symbols so that
        # character pairs can actually be discovered and merged.
        raw_word_freqs = Counter(words)

        word_freqs = Counter({
            " ".join(list(word)): freq
            for word, freq in raw_word_freqs.items()
        })

        if verbose:
            print(f"Unique words: {len(raw_word_freqs)}")

        # Initialize vocabulary with characters
        self._build_initial_vocab(word_freqs)

        # BPE merges
        num_merges = target_vocab - self._next_id
        for i in range(num_merges):
            if verbose and i % 100 == 0:
                print(f"  Merge {i}/{num_merges}, vocab size: {self.vocab_size}")

            pair = self._find_best_pair(word_freqs)
            if pair is None:
                if verbose:
                    print("No more pairs to merge")
                break

            self._merge_pair(pair, word_freqs)
            self.merges.append(pair)
            self.merge_ranks[pair] = len(self.merges) - 1

        self._initialized = True
        if verbose:
            print(f"Training complete. Final vocab size: {self.vocab_size}")

    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text using regex pattern."""
        return self.pattern.findall(text)

    def _build_initial_vocab(self, word_freqs: Counter) -> None:
        """Build initial vocabulary from BPE symbols."""
        chars = set()

        for word in word_freqs:
            for symbol in word.split():
                chars.add(symbol)

        for char in sorted(chars):
            if self._next_id >= self.config.vocab_size:
                break
            if char not in self.vocab:
                self.vocab[char] = self._next_id
                self.id_to_token[self._next_id] = char
                self._next_id += 1

        # Add byte fallback tokens if enabled
        if self.config.byte_fallback:
            for b in range(256):
                if self._next_id >= self.config.vocab_size:
                    break
                byte_token = f"<0x{b:02X}>"
                if byte_token not in self.vocab:
                    self.vocab[byte_token] = self._next_id
                    self.id_to_token[self._next_id] = byte_token
                    self._next_id += 1

    def _find_best_pair(self, word_freqs: Counter) -> Optional[Tuple[str, str]]:
        """Find the most frequent pair in the vocabulary."""
        pair_freqs = Counter()

        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq

        if not pair_freqs:
            return None

        # Filter by minimum frequency
        valid_pairs = {p: f for p, f in pair_freqs.items() if f >= self.config.min_frequency}
        if not valid_pairs:
            return None

        return max(valid_pairs, key=valid_pairs.get)

    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Counter) -> None:
        """Merge a pair in all words."""
        a, b = pair
        merged = a + b
        new_word_freqs = Counter()

        for word, freq in word_freqs.items():
            tokens = word.split()
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_freqs[" ".join(new_tokens)] = freq

        # Add merged token to vocabulary
        if merged not in self.vocab and self._next_id < self.config.vocab_size:
            self.vocab[merged] = self._next_id
            self.id_to_token[self._next_id] = merged
            self._next_id += 1

        word_freqs.clear()
        word_freqs.update(new_word_freqs)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")

        tokens = []
        words = self._pre_tokenize(text)

        for word in words:
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)

        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        return tokens

    @lru_cache(maxsize=100_000)
    def _encode_word(self, word: str) -> Tuple[int, ...]:
        """Encode a single word using BPE merges."""
        # Split into characters
        tokens = list(word)

        # Apply merges in order
        for merge in self.merges:
            a, b = merge
            merged = a + b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Byte fallback
                if self.config.byte_fallback:
                    for b in token.encode("utf-8"):
                        byte_token = f"<0x{b:02X}>"
                        if byte_token in self.vocab:
                            ids.append(self.vocab[byte_token])
                        else:
                            ids.append(self.unk_token_id)
                else:
                    ids.append(self.unk_token_id)

        return tuple(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Correctly reconstructs UTF-8 byte fallback sequences.
        """
        parts = []

        for id_ in ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]

                if skip_special_tokens and token in self.config.special_tokens:
                    continue

                parts.append(token)
            else:
                parts.append(self.id_to_token.get(self.unk_token_id, "<unk>"))

        output = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token in parts:
            match = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", token)

            if match:
                byte_buffer.append(int(match.group(1), 16))
            else:
                flush_bytes()
                output.append(token)

        flush_bytes()

        return "".join(output)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True, padding: bool = True, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        batch_ids = [self.encode(t, add_special_tokens) for t in texts]

        if max_length:
            batch_ids = [ids[:max_length] for ids in batch_ids]

        if padding:
            max_len = max(len(ids) for ids in batch_ids)
            attention_mask = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                ids.extend([self.pad_token_id] * pad_len)
                attention_mask.append([1] * (max_len - pad_len) + [0] * pad_len)
        else:
            attention_mask = [[1] * len(ids) for ids in batch_ids]

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        with open(path / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "pattern": self.config.pattern,
            "byte_fallback": self.config.byte_fallback,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load tokenizer from directory."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = TokenizerConfig(**config_dict)

        tokenizer = cls(config)

        # Load vocab
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._next_id = max(tokenizer.vocab.values()) + 1

        # Load merges
        with open(path / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ", 1)
                    tokenizer.merges.append((a, b))
                    tokenizer.merge_ranks[(a, b)] = len(tokenizer.merges) - 1

        tokenizer._initialized = True
        return tokenizer

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text: str, **kwargs) -> List[int]:
        return self.encode(text, **kwargs)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


class EntityTokenizer:
    """
    High-level tokenizer interface for the Entity model.
    Wraps BPETokenizer with additional utilities.
    """

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "EntityTokenizer":
        """Load tokenizer from directory."""
        return cls(BPETokenizer.load(path))

    @classmethod
    def train_new(cls, texts: List[str], vocab_size: int = 8192, **kwargs) -> "EntityTokenizer":
        """Train a new tokenizer."""
        config = TokenizerConfig(vocab_size=vocab_size, **kwargs)
        tokenizer = BPETokenizer(config)
        tokenizer.train(texts)
        return cls(tokenizer)

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(ids, **kwargs)

    def __call__(self, text: str, **kwargs) -> List[int]:
        return self.encode(text, **kwargs)

    def save(self, path: Union[str, Path]) -> None:
        self.tokenizer.save(path)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
