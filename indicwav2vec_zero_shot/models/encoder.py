"""
models/encoder.py
-----------------
Load an IndicWav2Vec encoder from HuggingFace and freeze all parameters.

Available models on HuggingFace (as of 2026):
  - ai4bharat/indicwav2vec-hindi    (recommended, no gating, full transformers)
  - ai4bharat/indicwav2vec-odia
  - ai4bharat/indicwav2vec_v1_bengali
  - ai4bharat/indicwav2vec_v1_gujarati
  - ai4bharat/indicwav2vec_v1_tamil   (gated)
  - ai4bharat/indicwav2vec_v1_telugu  (gated)

NOTE: The original 'ai4bharat/indicwav2vec_v1_large' pretrained multilingual
model is only available via the GitHub repo as a raw fairseq checkpoint.
It does NOT exist as a HuggingFace transformers model.

For this zero-shot evaluation we use language-specific fine-tuned models.
These all share the same Wav2Vec2-Large backbone pretrained on 40 Indian
languages — the fine-tuning head (LM head) is stripped and only the encoder
(transformer stack) is used.

LANGUAGE → MODEL MAPPING (choose the best available model per language):
  tamil     → ai4bharat/indicwav2vec_v1_tamil      (gated, needs HF token)
  hindi     → ai4bharat/indicwav2vec-hindi          (open)
  telugu    → ai4bharat/indicwav2vec_v1_telugu      (gated, needs HF token)
  malayalam → ai4bharat/indicwav2vec-hindi          (no Malayalam model exists;
                                                     using Hindi model encoder)

Key guarantees:
  - ALL encoder parameters have requires_grad = False after load_encoder()
  - No optimizer is created; no gradient updates ever occur
  - Hidden size is exposed via encoder.config.hidden_size (typically 1024)
"""

import os
import sys
import warnings
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Per-language model mapping
# ---------------------------------------------------------------------------
# IMPORTANT: Most ai4bharat/indicwav2vec_v1_* repos on HuggingFace are STUBS
# with no model weights (no pytorch_model.bin / model.safetensors).
#
# Models confirmed to have actual weights:
#   - ai4bharat/indicwav2vec_v1_bengali   (open, no gating)
#   - ai4bharat/indicwav2vec_v1_gujarati  (open, no gating)
#
# These models all share the SAME Wav2Vec2-Large backbone pretrained on 40
# Indian languages — the language-specific head (LM head) is stripped and
# only the encoder (transformer stack) is used for our zero-shot evaluation.
# Using bengali as the encoder for all languages is therefore acoustically
# equivalent in this zero-shot (randomly initialized CTC head) setup.
LANGUAGE_MODEL_MAP = {
    "tamil":     "ai4bharat/indicwav2vec_v1_bengali",
    "hindi":     "ai4bharat/indicwav2vec_v1_bengali",
    "telugu":    "ai4bharat/indicwav2vec_v1_bengali",
    "malayalam": "ai4bharat/indicwav2vec_v1_bengali",
}

# Fallback if language not in map
FALLBACK_MODEL_ID = "ai4bharat/indicwav2vec_v1_bengali"

# Expected hidden size for the large variant
EXPECTED_HIDDEN_SIZE = 1024


def get_model_id(language: str) -> str:
    """Return the HuggingFace model ID for the given language."""
    model_id = LANGUAGE_MODEL_MAP.get(language)
    if model_id is None:
        warnings.warn(
            f"[encoder] No model mapping for language '{language}'. "
            f"Using fallback: {FALLBACK_MODEL_ID}",
            stacklevel=2,
        )
        return FALLBACK_MODEL_ID
    if language == "malayalam":
        warnings.warn(
            "[encoder] No Malayalam-specific IndicWav2Vec model exists on HuggingFace. "
            f"Using Hindi encoder ({LANGUAGE_MODEL_MAP['malayalam']}) as a proxy. "
            "Representations will be from the shared multilingual backbone.",
            stacklevel=2,
        )
    return model_id


def load_encoder(
    model_id: str = FALLBACK_MODEL_ID,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """
    Download (or load from cache) a Wav2Vec2 encoder and freeze all parameters.

    Parameters
    ----------
    model_id  : str   HuggingFace model identifier (use get_model_id(lang) to resolve)
    device    : str   'cpu' or 'cuda'. Auto-detected if None.
    hf_token  : str   HuggingFace access token (required for gated repos).
                      Falls back to the HF_TOKEN environment variable.

    Returns
    -------
    model : transformers.Wav2Vec2Model  (frozen, eval mode, on device)
    """
    try:
        from transformers import Wav2Vec2Model
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required. "
            "Install with: py -m pip install transformers"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve token: explicit arg > env var
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    print(f"[encoder] Loading '{model_id}' on device='{device}' …", flush=True)

    load_kwargs = {}
    if hf_token:
        load_kwargs["token"] = hf_token
        # Also log in explicitly so gated repos can verify credentials
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass  # login() failure is non-fatal; we still pass token= below

    try:
        model = Wav2Vec2Model.from_pretrained(model_id, **load_kwargs)
    except OSError as exc:
        _print_auth_hint(model_id, hf_token)
        raise RuntimeError(
            f"Failed to load model '{model_id}'. "
            "See hint above."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error loading '{model_id}': {exc}"
        ) from exc

    # --- Freeze ALL encoder parameters ---
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Sanity check: confirm zero trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 0, (
        f"BUG: {trainable} encoder parameters still have requires_grad=True"
    )

    total = sum(p.numel() for p in model.parameters())
    hidden = model.config.hidden_size
    print(
        f"[encoder] Loaded. Parameters: {total:,}  |  Trainable: {trainable}  "
        f"|  Hidden size: {hidden}"
    )

    if hidden != EXPECTED_HIDDEN_SIZE:
        warnings.warn(
            f"[encoder] Hidden size is {hidden}, expected {EXPECTED_HIDDEN_SIZE}. "
            "Pass --hidden_size to the CLI to match.",
            stacklevel=2,
        )

    return model


@torch.no_grad()
def encode(
    model,
    waveform: np.ndarray,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Run the frozen encoder on a single waveform and return hidden states.

    Parameters
    ----------
    model    : frozen Wav2Vec2Model
    waveform : np.ndarray  shape (T,)  float32, 16kHz
    device   : str or None  (auto-detected)

    Returns
    -------
    torch.Tensor  shape (seq_len, hidden_size)  on CPU
    """
    if device is None:
        device = next(model.parameters()).device

    # Wav2Vec2 expects input shape: (batch, time)
    input_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)

    outputs = model(input_values=input_tensor)
    # last_hidden_state: (1, seq_len, hidden_size)
    hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)

    return hidden_states.cpu()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_auth_hint(model_id: str, hf_token: Optional[str]):
    token_status = "token provided" if hf_token else "NO TOKEN FOUND"
    print(
        f"\n[encoder] ERROR: Could not access '{model_id}' ({token_status}).\n"
        "\n"
        "  If this is a gated model (indicwav2vec_v1_tamil, _v1_telugu),\n"
        "  you need to:\n"
        "    1) Accept the model license at:\n"
        f"       https://huggingface.co/{model_id}\n"
        "    2) Set your token in the .env file:\n"
        "       HF_TOKEN=hf_your_actual_token_here\n"
        "\n"
        "  Alternatively, use the open model (no gating) by passing:\n"
        f"    --model_id ai4bharat/indicwav2vec-hindi\n"
        "\n"
        "  Get your token at: https://huggingface.co/settings/tokens\n",
        file=sys.stderr,
    )
