# IndicWav2Vec Zero-Shot ASR Evaluation System

A diagnostic evaluation pipeline for testing the **zero-shot representation quality** of [`ai4bharat/indicwav2vec_v1_large`](https://huggingface.co/ai4bharat/indicwav2vec_v1_large) across Tamil, Hindi, Telugu, and Malayalam.

> **This is not a production ASR system.** It is a controlled diagnostic tool.

---

## What "Zero-Shot ASR" Means Here

In this context, *zero-shot* means:

- The encoder (`indicwav2vec_v1_large`) is loaded **as-is**, with all weights frozen.
- No fine-tuning, no training, no gradient updates occur at any point.
- The CTC projection head is **randomly initialized** — it has never seen any text or audio labels.
- The system decodes speech by running **greedy argmax** over the random head's logits.

The question being answered is not *"can this model transcribe speech?"* but rather:  
**"Do the encoder's internal representations encode script-level and language-level structure?"**

---

## Why WER / CER Is Expected to Be High

Because the CTC head is random, the mapping from encoder features to characters is arbitrary. The decoder has no learned alignment between acoustic patterns and characters.

**CER values > 1.0 are expected and acceptable.** A CER of 2.5 simply means the hypothesis contains 2.5× as many character errors as the reference has characters — mostly due to random insertions and substitutions.

**Low WER/CER is NOT a success criterion for this system.**

---

## How to Interpret Results Correctly

| Signal | Interpretation |
|---|---|
| Output characters are in the correct script | ✅ Vocabulary covers the script block |
| Different languages produce different outputs | ✅ Encoder representations differ by language |
| CER ≈ 1.0–3.0 | ✅ Expected for random head |
| CER = 0.0 | 🚩 Suspicious — check vocab and decoder logic |
| Output is empty for all samples | ⚠️ Decoder may only emit blank — inspect logit distribution |
| System crashes | ❌ Bug — check logs |

---

## Acceptable Failure Modes

- **Very high CER/WER** — expected; random head produces random characters.
- **Empty hypotheses** — possible if the argmax only selects the blank token; this is a degenerate but valid CTC output.
- **Output is valid Unicode but semantically wrong** — expected; this is zero-shot.
- **Character distribution skewed** — random weights bias toward a subset of tokens; acceptable.

---

## Project Structure

```
indicwav2vec_zero_shot/
├── data/
│   ├── tamil/          ← Place 16kHz mono .wav files here
│   ├── hindi/
│   ├── telugu/
│   └── malayalam/
├── preprocess/
│   └── audio.py        ← WAV loading + validation
├── tokenizers/
│   └── build_char_vocab.py  ← Per-language Unicode character vocabs
├── models/
│   ├── encoder.py      ← Load + freeze IndicWav2Vec
│   └── ctc_probe.py    ← Random linear CTC head
├── decode/
│   └── greedy_ctc.py   ← Argmax → collapse → remove blank
├── eval/
│   ├── wer.py          ← CER + WER (edit distance, no deps)
│   └── qualitative_logger.py  ← Per-sample JSONL + summary table
├── run_zero_shot.py    ← CLI entry point
└── README.md
```

---

## Setup

### Requirements

```bash
pip install torch transformers soundfile numpy
```

### HuggingFace Authentication

`ai4bharat/indicwav2vec_v1_large` is a gated model. Authenticate before running:

```bash
# Option 1: CLI login
huggingface-cli login

# Option 2: Environment variable
set HF_TOKEN=your_token_here   # Windows PowerShell

# Option 3: Pass directly to the runner
python run_zero_shot.py --lang tamil --data_dir data/tamil --hf_token YOUR_TOKEN
```

---

## Usage

### Step 1 — Prepare audio

Place `.wav` files in the appropriate `data/{language}/` directory.

Requirements:
- Format: WAV
- Sample rate: **16 kHz**
- Channels: **mono**
- Encoding: float32
- Duration: 3–8 seconds (warning issued outside this range)

### Step 2 — Run evaluation

```powershell
cd c:\workspace\indicw2v\indicwav2vec_zero_shot

# Without references (no CER/WER — just decodes and logs)
python run_zero_shot.py --lang tamil --data_dir data/tamil

# With references (enables CER + WER)
python run_zero_shot.py `
    --lang tamil `
    --data_dir data/tamil `
    --ref_file data/tamil/refs.txt `
    --output_dir outputs/
```

### Reference file format (`refs.txt`)

Tab-separated, one entry per line:

```
audio_id_1<TAB>Tamil reference text here
audio_id_2<TAB>Another reference
```

### Step 3 — Inspect outputs

```
outputs/
├── tamil_log.jsonl      ← One JSON object per sample
└── tamil_summary.txt    ← Markdown summary table
```

Each line in the JSONL log:
```json
{
  "audio_id": "sample_001",
  "language": "tamil",
  "duration_sec": 4.12,
  "hypothesis": "மாமா மா",
  "reference": "நான் வீட்டிற்கு போகிறேன்",
  "cer": 1.8571,
  "wer": 2.0,
  "timestamp": "2026-02-27T11:30:00"
}
```

### Build vocabularies standalone

```powershell
python tokenizers/build_char_vocab.py --lang all
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--lang` | required | `tamil` / `hindi` / `telugu` / `malayalam` |
| `--data_dir` | required | Directory with `.wav` files |
| `--ref_file` | None | TSV references file (enables CER/WER) |
| `--output_dir` | `outputs/` | Where to write logs |
| `--hidden_size` | `1024` | Encoder hidden dim |
| `--seed` | `42` | CTC probe init seed |
| `--hf_token` | None | HuggingFace token |

---

## Why This Is Not Production ASR

| Property | This System | Production ASR |
|---|---|---|
| CTC head | Random weights | Fine-tuned on labelled data |
| Training data | None | Hours of transcribed speech |
| Language model | None | N-gram or neural LM |
| Decoding | Greedy argmax | Beam search + LM |
| Expected CER | >100% | 5–30% |
| Purpose | Diagnostic | Transcription |

---

## Reproducibility

- CTC probe initialized with `torch.manual_seed(42)` by default.
- Encoder weights are deterministic (loaded from HuggingFace cache).
- No stochastic operations during inference.
- Results are identical across runs given the same audio inputs and seed.
