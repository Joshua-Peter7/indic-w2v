"""
char_tokenizers/build_char_vocab.py
-------------------------------------
Build per-language character-level vocabularies for CTC decoding.

Each vocabulary is a JSON file mapping token → integer ID.
  - Index 0 is always reserved for the CTC blank token: '<blank>'
  - '|' is the space token
  - All Unicode characters in the language's script block are included

Usage (CLI):
    python tokenizers/build_char_vocab.py --lang tamil
    python tokenizers/build_char_vocab.py --lang all

Usage (library):
    from tokenizers.build_char_vocab import build_vocab, load_vocab
"""

import argparse
import json
import os
from typing import Dict

# ---------------------------------------------------------------------------
# Language → Unicode script block definitions
# ---------------------------------------------------------------------------
LANGUAGE_RANGES: Dict[str, list] = {
    "tamil":     [(0x0B80, 0x0BFF)],
    "hindi":     [(0x0900, 0x097F)],   # Devanagari
    "telugu":    [(0x0C00, 0x0C7F)],
    "malayalam": [(0x0D00, 0x0D7F)],
}

SUPPORTED_LANGUAGES = list(LANGUAGE_RANGES.keys())
BLANK_TOKEN = "<blank>"
SPACE_TOKEN = "|"

# Where vocabularies are saved (relative to this file's package root)
_VOCAB_DIR = os.path.dirname(os.path.abspath(__file__))


def _collect_script_chars(ranges: list) -> list:
    """
    Collect all assigned Unicode characters within the given (start, end) ranges.

    Only characters that Python's unicodedata considers valid (i.e. ord(ch) is
    a valid code point and the character is not a control character category 'Cc')
    are included.
    """
    import unicodedata

    chars = []
    for start, end in ranges:
        for codepoint in range(start, end + 1):
            try:
                ch = chr(codepoint)
            except ValueError:
                continue
            cat = unicodedata.category(ch)
            # Include letters, marks, numbers, punctuation in the script block.
            # Exclude pure control characters (Cc) and unassigned (Cn).
            if cat not in ("Cc", "Cn"):
                chars.append(ch)
    return chars


def build_vocab(language: str) -> Dict[str, int]:
    """
    Build and return the vocabulary dict for the given language.

    Token ordering:
        0       → '<blank>' (CTC blank)
        1       → '|'       (space)
        2..N    → script characters in Unicode order

    Parameters
    ----------
    language : str
        One of 'tamil', 'hindi', 'telugu', 'malayalam'.

    Returns
    -------
    dict mapping token (str) → integer ID
    """
    if language not in LANGUAGE_RANGES:
        raise ValueError(
            f"Unknown language '{language}'. "
            f"Supported: {SUPPORTED_LANGUAGES}"
        )

    script_chars = _collect_script_chars(LANGUAGE_RANGES[language])

    vocab: Dict[str, int] = {BLANK_TOKEN: 0, SPACE_TOKEN: 1}
    for idx, ch in enumerate(script_chars, start=2):
        # Avoid duplicate entries (shouldn't happen with clean ranges)
        if ch not in vocab:
            vocab[ch] = len(vocab)

    return vocab


def save_vocab(language: str, vocab: Dict[str, int], out_dir: str = None) -> str:
    """
    Serialize the vocabulary to a JSON file.

    Parameters
    ----------
    language : str
    vocab    : dict  token → id
    out_dir  : str   directory to write to (defaults to tokenizers/)

    Returns
    -------
    str  absolute path of written file
    """
    if out_dir is None:
        out_dir = _VOCAB_DIR

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{language}_vocab.json")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    return out_path


def load_vocab(language: str, vocab_dir: str = None) -> Dict[str, int]:
    """
    Load a previously saved vocabulary JSON for the given language.

    Parameters
    ----------
    language  : str
    vocab_dir : str  directory to look in (defaults to tokenizers/)

    Returns
    -------
    dict mapping token (str) → integer ID

    Raises
    ------
    FileNotFoundError if the vocab JSON does not exist.
    """
    if vocab_dir is None:
        vocab_dir = _VOCAB_DIR

    vocab_path = os.path.join(vocab_dir, f"{language}_vocab.json")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            f"Run: python tokenizers/build_char_vocab.py --lang {language}"
        )

    with open(vocab_path, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)

    return vocab


def get_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Return id → token mapping (inverse of vocab dict)."""
    return {v: k for k, v in vocab.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-language character vocabularies for CTC decoding."
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=SUPPORTED_LANGUAGES + ["all"],
        help="Language to build vocab for, or 'all' to build for every language.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Directory where vocab JSON files are written. "
            "Defaults to the tokenizers/ package directory."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    langs = SUPPORTED_LANGUAGES if args.lang == "all" else [args.lang]

    for lang in langs:
        vocab = build_vocab(lang)
        path = save_vocab(lang, vocab, out_dir=args.out_dir)
        print(
            f"[vocab] {lang:12s} → {len(vocab):4d} tokens  →  {path}"
        )


if __name__ == "__main__":
    main()
