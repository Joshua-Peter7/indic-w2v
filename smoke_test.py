"""Smoke test — runs without downloading the encoder."""
import sys, os, json, warnings, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

from char_tokenizers.build_char_vocab import build_vocab, get_inverse_vocab, save_vocab, SUPPORTED_LANGUAGES
from preprocess.audio import collect_wav_files, load_audio
from decode.greedy_ctc import greedy_decode, _collapse_and_remove_blank
from models.ctc_probe import build_ctc_probe
from eval.wer import compute_cer, compute_wer

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}  {detail}")
        errors.append(label)

# ------------------------------------------------------------------ #
# 1. Vocabulary
# ------------------------------------------------------------------ #
print("\n[1/5] Vocabulary Build")
vocab_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_tokenizers")
expected_sizes = {"tamil": (60, 90), "hindi": (100, 150), "telugu": (80, 130), "malayalam": (100, 140)}

for lang in SUPPORTED_LANGUAGES:
    vocab = build_vocab(lang)
    path = save_vocab(lang, vocab, out_dir=vocab_dir)
    inv  = get_inverse_vocab(vocab)

    check(f"{lang}: blank at index 0",   vocab.get("<blank>") == 0)
    check(f"{lang}: space at index 1",   vocab.get("|") == 1)
    check(f"{lang}: inv[0] == <blank>",  inv.get(0) == "<blank>")
    check(f"{lang}: JSON round-trip",    json.load(open(path, encoding="utf-8")) == vocab)
    lo, hi = expected_sizes[lang]
    check(f"{lang}: vocab size in ({lo},{hi})", lo < len(vocab) < hi, f"got {len(vocab)}")

# ------------------------------------------------------------------ #
# 2. Audio collector (empty dir)
# ------------------------------------------------------------------ #
print("\n[2/5] Audio Collector")
data_tamil = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tamil")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    files = collect_wav_files(data_tamil)
check("empty dir returns []",               files == [])
check("empty dir issues warning",           any("No .wav files" in str(x.message) for x in w))

# ------------------------------------------------------------------ #
# 3. Audio load (synthetic sine wave)
# ------------------------------------------------------------------ #
print("\n[3/5] Audio Load (synthetic 4s 440Hz sine)")
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    tmp_path = f.name

tone = np.sin(2 * np.pi * 440 * np.arange(16000 * 4) / 16000).astype(np.float32)
sf.write(tmp_path, tone, 16000)
audio = load_audio(tmp_path)
os.unlink(tmp_path)

check("sample_rate == 16000",              audio["sample_rate"] == 16000)
check("waveform dtype == float32",         audio["waveform"].dtype == np.float32)
check("duration ~4.0s",                    abs(audio["duration_sec"] - 4.0) < 0.05)
check("audio_id from filename",            isinstance(audio["audio_id"], str) and len(audio["audio_id"]) > 0)

# ------------------------------------------------------------------ #
# 4. CTC decode
# ------------------------------------------------------------------ #
print("\n[4/5] Greedy CTC Decode")
check("collapse+blank [2,2,0,2] -> [2,2]", _collapse_and_remove_blank([2,2,0,2]) == [2,2])
check("collapse+blank [0,0,0] -> []",       _collapse_and_remove_blank([0,0,0]) == [])
check("collapse+blank [] -> []",            _collapse_and_remove_blank([]) == [])
check("blank separates [1,0,1] -> [1,1]",  _collapse_and_remove_blank([1,0,1]) == [1,1])

for lang in SUPPORTED_LANGUAGES:
    vocab = build_vocab(lang)
    inv   = get_inverse_vocab(vocab)
    probe = build_ctc_probe(1024, vocab, seed=42)
    x     = torch.randn(30, 1024)
    with torch.no_grad():
        logits = probe(x)
    check(f"{lang}: probe output shape (30,{len(vocab)})", logits.shape == (30, len(vocab)))
    decoded = greedy_decode(logits, inv)
    check(f"{lang}: decode returns str",   isinstance(decoded, str))

# ------------------------------------------------------------------ #
# 5. CER / WER
# ------------------------------------------------------------------ #
print("\n[5/5] CER / WER")
check("CER('abc','abc') == 0.0",           compute_cer("abc","abc") == 0.0)
check("CER('abc','ab') == 1/3",            abs(compute_cer("abc","ab") - 1/3) < 1e-9)
check("CER('','') == 0.0",                 compute_cer("","") == 0.0)
check("CER('','x') == inf",               compute_cer("","x") == float("inf"))
check("WER('a b c','a b c') == 0.0",      compute_wer("a b c","a b c") == 0.0)
check("WER('a b c','a b') == 1/3",        abs(compute_wer("a b c","a b") - 1/3) < 1e-9)
check("WER('a b c','x y z') == 1.0",      abs(compute_wer("a b c","x y z") - 1.0) < 1e-9)

# ------------------------------------------------------------------ #
print()
if errors:
    print(f"FAILED: {len(errors)} checks failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("=== ALL CHECKS PASSED ===")
