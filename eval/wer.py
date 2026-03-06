"""
eval/wer.py
-----------
Character Error Rate (CER) and Word Error Rate (WER) computation.

Uses dynamic programming (Wagner-Fischer) for edit distance.
Does NOT use any external library — self-contained and dependency-free.

CER (primary metric):
    edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)

WER (secondary / informational):
    edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)

Both can exceed 1.0 in the presence of many insertions.
Results are NOT clamped and NOT averaged by default.
"""

from typing import List, Sequence, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Core: edit distance
# ---------------------------------------------------------------------------

def edit_distance(ref: Sequence[T], hyp: Sequence[T]) -> int:
    """
    Standard Levenshtein edit distance between two sequences.

    Costs: insertion=1, deletion=1, substitution=1.

    Parameters
    ----------
    ref, hyp : sequences (list, str, etc.)

    Returns
    -------
    int  minimum edit distance
    """
    m, n = len(ref), len(hyp)

    # Use two rows of DP to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def compute_cer(ref: str, hyp: str) -> float:
    """
    Character Error Rate.

    Parameters
    ----------
    ref : str  Reference (ground truth) text
    hyp : str  Hypothesis (predicted) text

    Returns
    -------
    float  CER in [0, ∞).  0.0 = perfect.  >1.0 = more errors than chars in ref.
           Returns 0.0 if ref is empty and hyp is also empty.
           Returns float('inf') if ref is empty but hyp is not.
    """
    ref_chars = list(ref)
    hyp_chars = list(hyp)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    dist = edit_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def compute_wer(ref: str, hyp: str) -> float:
    """
    Word Error Rate.

    Parameters
    ----------
    ref : str  Reference text (words separated by whitespace)
    hyp : str  Hypothesis text

    Returns
    -------
    float  WER in [0, ∞).
    """
    ref_words: List[str] = ref.split()
    hyp_words: List[str] = hyp.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    dist = edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


# ---------------------------------------------------------------------------
# Convenience: compute both at once
# ---------------------------------------------------------------------------

def compute_metrics(ref: str, hyp: str) -> dict:
    """
    Compute both CER and WER, returning a dict.

    Returns
    -------
    dict with keys 'cer' and 'wer'
    """
    return {
        "cer": compute_cer(ref, hyp),
        "wer": compute_wer(ref, hyp),
    }
