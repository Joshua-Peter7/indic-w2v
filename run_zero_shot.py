"""
run_zero_shot.py
----------------
CLI entry point for the zero-shot IndicWav2Vec ASR evaluation pipeline.

HuggingFace token is loaded automatically from the .env file in this
directory. Set HF_TOKEN=your_token_here in .env before running.
Alternatively, use --hf_token or the HF_TOKEN environment variable.

Usage
-----
    python run_zero_shot.py --lang tamil --data_dir data/tamil

With references (enables CER/WER):
    python run_zero_shot.py \\
        --lang tamil \\
        --data_dir data/tamil \\
        --ref_file data/tamil/refs.txt \\
        --output_dir outputs/

Arguments
---------
    --lang        One of: tamil | hindi | telugu | malayalam
    --data_dir    Directory containing .wav files (16kHz, mono)
    --ref_file    Optional TSV: audio_id<TAB>reference text (per line)
    --output_dir  Where to write JSONL logs (default: outputs/)
    --hidden_size Encoder hidden dimension (default: 1024 for large variant)
    --seed        Random seed for CTC probe initialization (default: 42)
    --hf_token    HuggingFace access token (falls back to HF_TOKEN env var)

Pipeline
--------
  .wav files → preprocess → IndicWav2Vec encoder (frozen) →
  random CTC linear head → greedy CTC decode → log + summarize
"""

import argparse
import os
import sys

import torch


# ---------------------------------------------------------------------------
# Load .env file before anything else (provides HF_TOKEN without python-dotenv)
# ---------------------------------------------------------------------------
def _load_dotenv():
    """Parse a simple KEY=VALUE .env file and inject into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if not already in environment (env var wins over .env)
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


# ---------------------------------------------------------------------------
# Imports — explicit relative paths so the script works from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess.audio import load_audio, collect_audio_files
from char_tokenizers.build_char_vocab import build_vocab, save_vocab, get_inverse_vocab
from models.encoder import load_encoder, encode, get_model_id
from models.ctc_probe import build_ctc_probe
from decode.greedy_ctc import greedy_decode
from eval.wer import compute_cer, compute_wer
from eval.qualitative_logger import QualitativeLogger, load_references

SUPPORTED_LANGUAGES = ["tamil", "hindi", "telugu", "malayalam"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot ASR evaluation using IndicWav2Vec (frozen) + "
            "random CTC head + greedy decoding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=SUPPORTED_LANGUAGES,
        help="Language to evaluate.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory with .wav audio files (16kHz, mono).",
    )
    parser.add_argument(
        "--ref_file",
        default=None,
        help=(
            "Optional TSV file: 'audio_id<TAB>reference text' per line. "
            "If omitted, CER/WER are not computed."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write JSONL logs and summaries (default: outputs/).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Encoder hidden size (default: 1024 for indicwav2vec_v1_large).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CTC probe initialization (default: 42).",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help=(
            "HuggingFace access token. Falls back to the HF_TOKEN "
            "environment variable if not provided."
        ),
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help=(
            "Override the HuggingFace model ID. "
            "Defaults to the per-language model (see models/encoder.py). "
            "Use 'ai4bharat/indicwav2vec-hindi' for an open model with no gating."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # Resolve which model to use for this language
    model_id = args.model_id or get_model_id(args.lang)

    print("=" * 60)
    print(f"  Zero-Shot IndicWav2Vec ASR Evaluation")
    print(f"  Language   : {args.lang}")
    print(f"  Model      : {model_id}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build / load vocabulary
    # ------------------------------------------------------------------
    print(f"\n[step 1/5] Building character vocabulary for '{args.lang}' …")
    vocab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_tokenizers")
    vocab = build_vocab(args.lang)
    save_vocab(args.lang, vocab, out_dir=vocab_dir)
    inv_vocab = get_inverse_vocab(vocab)
    print(f"  Vocabulary size: {len(vocab)} tokens  (blank + space + script chars)")

    # ------------------------------------------------------------------
    # 2. Load frozen encoder
    # ------------------------------------------------------------------
    print(f"\n[step 2/5] Loading encoder …")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    encoder = load_encoder(model_id=model_id, hf_token=hf_token)
    device = next(encoder.parameters()).device

    # ------------------------------------------------------------------
    # 3. Build random CTC probe
    # ------------------------------------------------------------------
    print(f"\n[step 3/5] Initializing random CTC probe …")
    probe = build_ctc_probe(
        hidden_size=args.hidden_size,
        vocab=vocab,
        seed=args.seed,
    )
    probe.to(device)

    # ------------------------------------------------------------------
    # 4. Collect audio files + optional references
    # ------------------------------------------------------------------
    print(f"\n[step 4/5] Collecting audio files from '{args.data_dir}' …")
    wav_files = collect_audio_files(args.data_dir)
    print(f"  Found {len(wav_files)} audio file(s) (WAV/MP3/FLAC).")

    if len(wav_files) == 0:
        print(
            f"\n[warning] No audio files to process. "
            f"Place .wav files (16kHz, mono) in '{args.data_dir}' and re-run."
        )
        return

    references: dict = {}
    if args.ref_file:
        print(f"  Loading references from '{args.ref_file}' …")
        references = load_references(args.ref_file)
        print(f"  Loaded {len(references)} reference(s).")

    # ------------------------------------------------------------------
    # 5. Inference loop
    # ------------------------------------------------------------------
    print(f"\n[step 5/5] Running inference …\n")

    os.makedirs(args.output_dir, exist_ok=True)

    with QualitativeLogger(args.lang, args.output_dir) as logger:
        for wav_path in wav_files:
            audio_id = os.path.splitext(os.path.basename(wav_path))[0]
            try:
                # --- Preprocess ---
                audio = load_audio(wav_path)
                waveform = audio["waveform"]
                duration_sec = audio["duration_sec"]

                # --- Encode (frozen, no grad) ---
                with torch.no_grad():
                    hidden_states = encode(encoder, waveform)  # (T, H)

                    # --- CTC probe ---
                    logits = probe(hidden_states.to(device))   # (T, V)

                # --- Greedy decode ---
                hypothesis = greedy_decode(logits.cpu(), inv_vocab)

                # --- Metrics ---
                reference = references.get(audio_id)
                cer = compute_cer(reference, hypothesis) if reference else None
                wer = compute_wer(reference, hypothesis) if reference else None

                # --- Log ---
                logger.log_sample(
                    audio_id=audio_id,
                    hypothesis=hypothesis,
                    duration_sec=duration_sec,
                    reference=reference,
                    cer=cer,
                    wer=wer,
                )

                _print_sample(audio_id, duration_sec, reference, hypothesis, cer, wer)

            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"  [ERROR] {audio_id}: {exc}", file=sys.stderr)
                # Log the failure so it is visible in the JSONL
                logger.log_sample(
                    audio_id=audio_id,
                    hypothesis=f"[ERROR: {exc}]",
                    duration_sec=0.0,
                )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        logger.write_summary()

    print("\n[done] Evaluation complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_sample(
    audio_id: str,
    duration_sec: float,
    reference,
    hypothesis: str,
    cer,
    wer,
) -> None:
    print(f"  [{audio_id}]  duration={duration_sec:.2f}s")
    if reference:
        print(f"    ref : {reference}")
    print(f"    hyp : {hypothesis or '(empty)'}")
    if cer is not None:
        print(f"    CER : {cer:.4f}   WER : {wer:.4f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
