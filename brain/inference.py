"""
Entity Inference Engine

High-level inference interface for the Entity model.
Handles tokenization, generation, and post-processing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import EntityTransformer, EntityConfig
from .tokenizer import EntityTokenizer, BPETokenizer
from .language import clean_assistant_response


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling."""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    do_sample: bool = True
    eos_token_id: int = 3
    pad_token_id: int = 0
    # Advanced
    min_p: float = 0.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    # Stop sequences
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str
    token_ids: List[int]
    input_token_count: int
    output_token_count: int
    finish_reason: str  # "eos", "length", "stop_sequence"
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """
    High-level inference engine for the Entity model.

    Handles:
    - Model loading and device management
    - Tokenization
    - Text generation with various sampling strategies
    - Batch inference
    - Streaming generation
    """

    def __init__(
        self,
        model: EntityTransformer,
        tokenizer: EntityTokenizer,
        device: Optional[Union[str, torch.device]] = None,
        default_sampling: Optional[SamplingConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_sampling = default_sampling or SamplingConfig()

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Compile if available (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass  # Compile failed, use eager mode

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> "InferenceEngine":
        """
        Load inference engine from checkpoint.

        Args:
            model_path: Path to model checkpoint (.pt or .bin)
            tokenizer_path: Path to tokenizer directory
            device: Device to load model on
            config_path: Optional path to config.json
        """
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)

        # Load config
        if config_path:
            config = EntityConfig.load(config_path)
        else:
            config_path = model_path.parent / "config.json"
            if config_path.exists():
                config = EntityConfig.load(config_path)
            else:
                config = EntityConfig()

        # Load tokenizer
        tokenizer = EntityTokenizer.from_pretrained(tokenizer_path)

        # Update config with tokenizer vocab size
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.unk_token_id = tokenizer.unk_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id

        # Create model
        model = EntityTransformer(config)

        # Load weights
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return cls(model, tokenizer, device)

    def generate(
        self,
        prompt: str,
        sampling: Optional[SamplingConfig] = None,
        return_metadata: bool = False,
    ) -> Union[str, GenerationResult]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            sampling: Sampling configuration (uses default if None)
            return_metadata: Whether to return full GenerationResult

        Returns:
            Generated text or GenerationResult
        """
        sampling = sampling or self.default_sampling

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # Dialogue and retrieved memories can exceed the model context window.
        # Preserve the newest tokens, which contain the current user turn.
        max_prompt_tokens = max(1, self.model.config.max_seq_len - sampling.max_new_tokens)
        if len(input_ids) > max_prompt_tokens:
            input_ids = input_ids[-max_prompt_tokens:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=sampling.max_new_tokens,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                repetition_penalty=sampling.repetition_penalty,
                eos_token_id=sampling.eos_token_id,
                pad_token_id=sampling.pad_token_id,
                do_sample=sampling.do_sample,
            )

        # Decode
        output_ids = output_ids[0].tolist()
        generated_ids = output_ids[len(input_ids):]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Check finish reason
        finish_reason = "length"
        if sampling.eos_token_id in generated_ids:
            finish_reason = "eos"
        for stop_seq in sampling.stop_sequences:
            if stop_seq in text:
                finish_reason = "stop_sequence"
                text = text[:text.index(stop_seq)]
                break

        text = clean_assistant_response(text)

        if return_metadata:
            return GenerationResult(
                text=text,
                token_ids=generated_ids,
                input_token_count=len(input_ids),
                output_token_count=len(generated_ids),
                finish_reason=finish_reason,
                metadata={
                    "sampling_config": sampling.__dict__,
                    "model_config": self.model.config.to_dict(),
                },
            )
        return text

    def generate_batch(
        self,
        prompts: List[str],
        sampling: Optional[SamplingConfig] = None,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        return [self.generate(p, sampling, return_metadata=True) for p in prompts]

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        sampling: Optional[SamplingConfig] = None,
    ):
        """
        Stream generation token by token.

        Yields:
            Tuple of (token_id, token_text, is_finished)
        """
        sampling = sampling or self.default_sampling

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        max_prompt_tokens = max(1, self.model.config.max_seq_len - sampling.max_new_tokens)
        if len(input_ids) > max_prompt_tokens:
            input_ids = input_ids[-max_prompt_tokens:]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        past_key_values = None
        generated_ids = []

        for step in range(sampling.max_new_tokens):
            # Forward pass
            outputs = self.model.forward(
                input_tensor if past_key_values is None else input_tensor[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if sampling.repetition_penalty != 1.0:
                all_ids = input_ids + generated_ids
                for token_id in set(all_ids):
                    logits[0, token_id] /= sampling.repetition_penalty

            # Temperature
            if sampling.temperature > 0:
                logits = logits / sampling.temperature

            # Top-k
            if sampling.top_k > 0:
                top_k = min(sampling.top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
                logits[indices_to_remove] = float("-inf")

            # Top-p
            if sampling.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > sampling.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            if sampling.do_sample and sampling.temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated_ids.append(token_id)
            input_tensor = next_token

            # Decode token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            # Check stop conditions
            is_finished = False
            if token_id == sampling.eos_token_id:
                is_finished = True
            for stop_seq in sampling.stop_sequences:
                if stop_seq in "".join(self.tokenizer.decode([tid], skip_special_tokens=True) for tid in generated_ids):
                    is_finished = True
                    break

            yield token_id, token_text, is_finished

            if is_finished:
                break

    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get raw logits for a prompt (for analysis/debugging)."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.forward(input_tensor)
            logits = outputs["logits"]

        return logits[0, -1, :]  # Last token logits

    def get_embeddings(self, prompt: str, layer: int = -1) -> torch.Tensor:
        """Get hidden states from a specific layer."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Hook to capture hidden states
        hidden_states = []

        def hook(module, input, output):
            hidden_states.append(output[0] if isinstance(output, tuple) else output)

        handle = self.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            self.model.forward(input_tensor)

        handle.remove()
        return hidden_states[0] if hidden_states else None

    def estimate_perplexity(self, texts: List[str], batch_size: int = 4) -> float:
        """Estimate perplexity on a list of texts."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer.encode_batch(batch_texts, padding=True)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.forward(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]

            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            loss = (loss * shift_mask.view(-1)).sum()
            tokens = shift_mask.sum()

            total_loss += loss.item()
            total_tokens += tokens.item()

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    def __repr__(self) -> str:
        return f"InferenceEngine(model={self.model.config.model_name}, device={self.device})"
