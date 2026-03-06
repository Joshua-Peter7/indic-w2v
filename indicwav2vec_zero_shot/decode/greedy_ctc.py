"""
decode/greedy_ctc.py
--------------------
Greedy CTC decoding: argmax → collapse repeats → remove blank → map to chars.

No language model. No beam search. Deterministic and simple.

CTC Decoding rules:
  1. At each time step t, take the token with the highest logit (argmax).
  2. Collapse consecutive identical token IDs into one.
  3. Remove the blank token (index 0).
  4. Map remaining IDs to characters using the inverse vocabulary.
  5. Join and return the raw output string (no normalization applied).
"""

from typing import Dict, List

import torch


BLANK_ID = 0  # CTC blank is always at index 0 by convention in this project


def greedy_decode(
    logits: torch.Tensor,
    inv_vocab: Dict[int, str],
) -> str:
    """
    Decode a sequence of logits using greedy CTC.

    Parameters
    ----------
    logits   : torch.Tensor  shape (T, vocab_size+1) — raw (unnormalized) logits
    inv_vocab: dict  id → token  (as returned by build_char_vocab.get_inverse_vocab)

    Returns
    -------
    str  Decoded text. May be empty if only blanks were produced.
         Space token '|' is replaced with a space character ' '.
         Raw output — no additional normalization.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"[greedy_ctc] Expected logits shape (T, V), got {tuple(logits.shape)}"
        )

    # Step 1: Argmax at each timestep
    token_ids: torch.Tensor = logits.argmax(dim=-1)  # shape (T,)

    # Step 2 & 3: Collapse repeats and remove blank
    collapsed: List[int] = _collapse_and_remove_blank(token_ids.tolist())

    # Step 4: Map IDs → characters
    chars: List[str] = []
    for tid in collapsed:
        token = inv_vocab.get(tid, "")
        if token == "|":
            chars.append(" ")
        elif token:
            chars.append(token)

    # Step 5: Join
    return "".join(chars)


def _collapse_and_remove_blank(token_ids: List[int]) -> List[int]:
    """
    Collapse consecutive duplicate token IDs and remove blank (id=0).

    Example:
        [0, 3, 3, 0, 5, 0, 5] → after collapse: [0, 3, 0, 5, 0, 5]
                                → after remove blank: [3, 5, 5]

    Note: duplicates are only collapsed when they are consecutive.
    A sequence like [5, 0, 5] becomes [5, 5] after removing blank, which
    is correct CTC behaviour (blank acts as a separator allowing repeated chars).
    """
    collapsed: List[int] = []
    prev = None
    for tid in token_ids:
        if tid != prev:
            collapsed.append(tid)
            prev = tid

    return [tid for tid in collapsed if tid != BLANK_ID]


def decode_batch(
    logits_list: List[torch.Tensor],
    inv_vocab: Dict[int, str],
) -> List[str]:
    """
    Decode a list of logit tensors.

    Parameters
    ----------
    logits_list : list of Tensor, each shape (T_i, V)
    inv_vocab   : dict  id → token

    Returns
    -------
    list of str
    """
    return [greedy_decode(logits, inv_vocab) for logits in logits_list]
