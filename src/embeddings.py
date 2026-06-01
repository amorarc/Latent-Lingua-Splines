"""
Extracts static token embeddings from the embedding layer of a HuggingFace LM.

These are the *lookup-table* embeddings (before any transformer layers), so
each token maps to one fixed vector regardless of context.  The trajectory of
a phrase is the ordered sequence of those vectors — one point per token.
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import NamedTuple


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class PhraseTrajectory(NamedTuple):
    tokens: list[str]          # decoded token strings
    token_ids: list[int]
    embeddings: np.ndarray     # shape (n_tokens, hidden_size)


class EmbeddingExtractor:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        quantization: dict | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = self._build_bnb_config(quantization)
        is_quantized = bnb_config is not None

        print(f"Loading tokenizer and model: {model_name}")
        if is_quantized:
            bits = quantization["bits"]
            print(f"  Quantization: {bits}-bit (bitsandbytes)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs: dict = {}
        if is_quantized:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        model.eval()

        self._embedding_layer = self._extract_embedding_layer(model)

        # bitsandbytes places the model via device_map; don't move it again
        if not is_quantized:
            self._embedding_layer.to(self.device)

        self.hidden_size: int = self._embedding_layer.weight.shape[1]
        print(
            f"  Embedding layer: {self._embedding_layer.weight.shape[0]} tokens "
            f"× {self.hidden_size} dims"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bnb_config(quantization: dict | None) -> BitsAndBytesConfig | None:
        if not quantization or not quantization.get("enabled", False):
            return None

        bits = quantization.get("bits", 4)
        if bits not in (4, 8):
            raise ValueError(f"quantization.bits must be 4 or 8, got {bits}")

        if bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)

        compute_dtype = _DTYPE_MAP.get(
            quantization.get("compute_dtype", "float16"), torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quantization.get("double_quant", True),
        )

    @staticmethod
    def _extract_embedding_layer(model: torch.nn.Module) -> torch.nn.Embedding:
        """
        Walk common attribute paths to find the token embedding table.
        Works for BERT, RoBERTa, GPT-2, T5, LLaMA, Mistral, Qwen, and most
        encoder/decoder architectures available on HuggingFace.
        """
        candidates = [
            # BERT / RoBERTa / DistilBERT
            lambda m: m.embeddings.word_embeddings,
            # GPT-2 / GPT-Neo / GPT-J
            lambda m: m.transformer.wte,
            # T5 encoder
            lambda m: m.encoder.embed_tokens,
            # T5 shared (encoder + decoder share the same table)
            lambda m: m.shared,
            # LLaMA / Mistral / Qwen / Falcon via CausalLM wrapper
            lambda m: m.model.embed_tokens,
            # Generic "embed_tokens"
            lambda m: m.embed_tokens,
            # Generic "word_embeddings"
            lambda m: m.word_embeddings,
        ]
        for fn in candidates:
            try:
                layer = fn(model)
                if isinstance(layer, torch.nn.Embedding):
                    return layer
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate the token embedding layer automatically.\n"
            "Please open src/embeddings.py and add the attribute path for "
            f"model '{type(model).__name__}'."
        )

    def _ids_to_embeddings(self, token_ids: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            vecs = self._embedding_layer(token_ids.to(self.device))
        # cast to float32 so numpy conversion works regardless of model dtype
        return vecs.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def phrase_trajectory(self, phrase: str) -> PhraseTrajectory:
        """Return the ordered embedding vectors for every token in *phrase*."""
        encoding = self.tokenizer(
            phrase,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = encoding["input_ids"].squeeze(0)  # (n_tokens,)
        embeddings = self._ids_to_embeddings(ids)  # (n_tokens, hidden_size)
        tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        return PhraseTrajectory(
            tokens=tokens,
            token_ids=ids.tolist(),
            embeddings=embeddings,
        )

    def corpus_embeddings(self, phrases: list[str]) -> tuple[np.ndarray, list[str]]:
        """
        Collect all token embeddings from a list of phrases.

        Returns:
            embeddings: (N_total_tokens, hidden_size)
            tokens:     list of N_total_tokens decoded token strings
        """
        all_vecs: list[np.ndarray] = []
        all_tokens: list[str] = []

        for phrase in phrases:
            traj = self.phrase_trajectory(phrase)
            all_vecs.append(traj.embeddings)
            all_tokens.extend(traj.tokens)

        return np.vstack(all_vecs), all_tokens

    def full_vocab_embeddings(self) -> np.ndarray:
        """Return the entire embedding matrix (vocab_size, hidden_size)."""
        with torch.no_grad():
            return self._embedding_layer.weight.cpu().float().numpy()
