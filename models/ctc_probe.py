"""
models/ctc_probe.py
-------------------
Random linear CTC projection head.

This is a single nn.Linear layer with RANDOM initialization only.
No pretrained weights are loaded. This is intentional — the purpose of
this project is to evaluate zero-shot encoder representations, not to
produce sensible transcriptions.

Architecture:
    hidden_states (T, hidden_size)  →  logits (T, vocab_size + 1)

The +1 accounts for the CTC blank token (always at index 0).
"""

import torch
import torch.nn as nn


class CTCProbe(nn.Module):
    """
    Single randomly initialized linear layer for CTC decoding.

    Parameters
    ----------
    hidden_size : int   Dimensionality of encoder output (e.g. 1024)
    vocab_size  : int   Number of tokens excluding blank
    seed        : int   Manual seed for reproducibility (default: 42)
    """

    def __init__(self, hidden_size: int, vocab_size: int, seed: int = 42):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_size = vocab_size + 1  # +1 for blank at index 0

        # Deterministic random initialization
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.projection = nn.Linear(hidden_size, self.output_size)

        print(
            f"[ctc_probe] Initialized random CTC head: "
            f"{hidden_size} → {self.output_size} "
            f"(vocab_size={vocab_size} + 1 blank)  seed={seed}"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor  shape (T, hidden_size) or (B, T, hidden_size)

        Returns
        -------
        torch.Tensor  shape (T, output_size) or (B, T, output_size)
            Raw logits, NOT softmax/log_softmax.
        """
        return self.projection(hidden_states)


def build_ctc_probe(hidden_size: int, vocab: dict, seed: int = 42) -> CTCProbe:
    """
    Convenience constructor: derives vocab_size from a vocab dict.

    Parameters
    ----------
    hidden_size : int
    vocab       : dict  token → id  (as returned by build_char_vocab.build_vocab)
    seed        : int

    Returns
    -------
    CTCProbe (eval mode, no grad)
    """
    # vocab_size = total tokens - 1 (exclude the blank token itself)
    vocab_size = len(vocab) - 1  # blank is already in the vocab at index 0
    probe = CTCProbe(hidden_size=hidden_size, vocab_size=vocab_size, seed=seed)
    probe.eval()
    return probe
