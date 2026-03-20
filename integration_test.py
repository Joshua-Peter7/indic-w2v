"""Integration test: MP3 loading + dataset scan + .env detection."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess.audio import load_audio, collect_audio_files
PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS}  {label}")
    else
        print(f"  {FAIL}  {label}  {detail}")
        errors.append(label)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 1. Dataset scan
# ------------------------------------------------------------------
print("\n[1/3] Dataset scan")
for lang in ["tamil", "hindi", "telugu", "malayalam"]:
    d = os.path.join(ROOT, "data", lang)
    files = collect_audio_files(d)
    check(f"{lang}: 1000 audio files found", len(files) == 1000, f"got {len(files)}")
    check(f"{lang}: all are .mp3", all(f.endswith(".mp3") for f in files))

# ------------------------------------------------------------------
# 2. Real MP3 load (one file per language)
# ------------------------------------------------------------------
print("\n[2/3] MP3 loading (one sample per language)")
for lang in ["tamil", "hindi", "telugu", "malayalam"]:
    d = os.path.join(ROOT, "data", lang)
    sample = sorted(os.listdir(d))[0]  # first file alphabetically
    path = os.path.join(d, sample)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        audio = load_audio(path)
    check(f"{lang}: sample_rate == 16000", audio["sample_rate"] == 16000)
    check(f"{lang}: waveform is 1D", audio["waveform"].ndim == 1)
    check(f"{lang}: dtype == float32", str(audio["waveform"].dtype) == "float32")
    check(f"{lang}: duration > 0", audio["duration_sec"] > 0)
    print(f"         audio_id={audio['audio_id']}  dur={audio['duration_sec']:.2f}s")

# ------------------------------------------------------------------
# 3. .env token structure
# ------------------------------------------------------------------
print("\n[3/3] .env file")
env_path = os.path.join(ROOT, ".env")
check(".env file exists", os.path.isfile(env_path))
if os.path.isfile(env_path):
    keys = {}
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            keys[k.strip()] = v.strip()
    check("HF_TOKEN key present in .env", "HF_TOKEN" in keys)
    is_placeholder = keys.get("HF_TOKEN", "").startswith("hf_PASTE")
    print(f"  HF_TOKEN value is{'  PLACEHOLDER (expected)' if is_placeholder else ': REAL TOKEN SET'}")

print()
if errors:
    print(f"FAILED: {len(errors)} check(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("=== ALL INTEGRATION TESTS PASSED ===")
