"""
preprocess/audio.py
-------------------
Audio loading and validation for the zero-shot ASR pipeline.

Supports:
  - WAV  (via soundfile — fast, exact)
  - MP3  (via librosa  — decodes and resamples to 16kHz mono automatically)

Responsibilities:
  - Load audio file from disk (WAV or MP3)
  - Validate: mono, 16kHz sample rate, float32
  - Warn on duration outside 3–8s range (does not reject)
  - Return a dict with waveform, sample_rate, duration_sec, audio_id
"""

import os
import warnings
from typing import Dict, Any

import numpy as np

# Constants
REQUIRED_SAMPLE_RATE = 16_000
MIN_DURATION_SEC = 30.0
MAX_DURATION_SEC = 40.0

# Supported extensions
WAV_EXT  = {".wav"}
MP3_EXT  = {".mp3"}
FLAC_EXT = {".flac"}
SUPPORTED_EXTS = WAV_EXT | MP3_EXT | FLAC_EXT


def load_audio(audio_path: str) -> Dict[str, Any]:
    """
    Load an audio file (WAV, MP3, or FLAC) and validate properties.

    For MP3/FLAC:
      - Decoded and resampled to 16kHz via librosa
      - Mixed down to mono automatically

    For WAV:
      - Loaded via soundfile (faster, no resampling)
      - Must already be 16kHz mono (error if not)

    Parameters
    ----------
    audio_path : str
        Path to an audio file (.wav, .mp3, or .flac).

    Returns
    -------
    dict with keys:
        waveform     : np.ndarray  shape (T,)  dtype float32
        sample_rate  : int         always 16000
        duration_sec : float
        audio_id     : str         basename without extension

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : unsupported format or WAV is not 16kHz
    RuntimeError      : decoding failure
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTS)}"
        )

    if ext in WAV_EXT:
        waveform, sample_rate = _load_wav(audio_path)
    else:
        # MP3 or FLAC: use librosa (handles decoding + resample + mono)
        waveform, sample_rate = _load_with_librosa(audio_path)

    duration_sec = len(waveform) / sample_rate

    if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
        warnings.warn(
            f"[audio] '{os.path.basename(audio_path)}' duration {duration_sec:.2f}s "
            f"is outside recommended [{MIN_DURATION_SEC}s, {MAX_DURATION_SEC}s]. "
            "Processing continues.",
            stacklevel=2,
        )

    audio_id = os.path.splitext(os.path.basename(audio_path))[0]

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "audio_id": audio_id,
    }


def _load_wav(wav_path: str):
    """Load a WAV file using soundfile. Requires 16kHz sample rate."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Install soundfile: pip install soundfile") from exc

    try:
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"soundfile failed on '{wav_path}': {exc}") from exc

    # Multi-channel: average to mono
    if waveform.ndim == 2:
        warnings.warn(
            f"[audio] '{wav_path}' has {waveform.shape[1]} channels; "
            "averaging to mono.",
            stacklevel=3,
        )
        waveform = waveform.mean(axis=1)

    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"[audio] WAV '{wav_path}' has sample rate {sample_rate} Hz; "
            f"expected {REQUIRED_SAMPLE_RATE} Hz. "
            "Resample the file, or use an MP3 — librosa auto-resamples."
        )

    return waveform.astype(np.float32), sample_rate


def _load_with_librosa(audio_path: str):
    """
    Load MP3/FLAC using librosa.
    Automatically: decodes → resamples to 16kHz → mixes to mono.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Install librosa to read MP3/FLAC: pip install librosa"
        ) from exc

    try:
        # sr=REQUIRED_SAMPLE_RATE forces resample; mono=True mixes channels
        waveform, sample_rate = librosa.load(
            audio_path,
            sr=REQUIRED_SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"librosa failed to load '{audio_path}': {exc}\n"
            "Ensure ffmpeg is installed for MP3 support: https://ffmpeg.org"
        ) from exc

    return waveform, sample_rate


def collect_audio_files(data_dir: str) -> list:
    """
    Return sorted list of absolute paths to all supported audio files in data_dir.

    Accepts WAV, MP3, and FLAC. Non-recursive (top-level only).

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    list of str (absolute paths), sorted by filename
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

    if not files:
        warnings.warn(
            f"[audio] No audio files found in '{data_dir}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTS)}",
            stacklevel=2,
        )

    return files


# Keep old name as alias for backward compatibility
collect_wav_files = collect_audio_files
