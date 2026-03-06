"""
live_stt.py  —  Real-Time Speech-to-Text using OpenAI Whisper
=============================================================

Speak into your microphone. Whisper will automatically:
  - Detect the language you're speaking (Tamil, Hindi, English, etc.)
  - Transcribe what you say in real time

Usage
-----
    # Auto-detect language (default)
    py live_stt.py

    # Pin to Tamil
    py live_stt.py --lang ta

    # Pin to Hindi
    py live_stt.py --lang hi

    # Use a larger/smaller model  (tiny | base | small | medium | large-v3)
    py live_stt.py --model medium

    # Translate everything to English
    py live_stt.py --translate

Controls
--------
    Press  Ctrl+C  to stop.
"""

import argparse
import queue
import sys
import wave
import tempfile
import os
import time
import threading

import numpy as np
import sounddevice as sd
import whisper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LANGUAGE_NAMES = {
    "ta": "Tamil", "hi": "Hindi", "te": "Telugu",
    "ml": "Malayalam", "en": "English", "kn": "Kannada",
}

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live STT with Whisper")
    p.add_argument("--model",     default="small",  help="Whisper model size (default: small)")
    p.add_argument("--lang",      default=None,      help="Language code e.g. 'ta' for Tamil. Auto-detect if omitted.")
    p.add_argument("--translate", action="store_true", help="Translate to English instead of transcribe.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colours (ANSI)
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
GREY   = "\033[90m"
RED    = "\033[91m"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"\n{BOLD}{'='*58}")
    print("  🎙️  Live Speech-to-Text  —  Whisper")
    print(f"{'='*58}{RESET}")
    print(f"  Model     : {BOLD}{args.model}{RESET}")

    lang_display = LANGUAGE_NAMES.get(args.lang, args.lang) if args.lang else "Auto-detect"
    print(f"  Language  : {CYAN}{lang_display}{RESET}")
    print(f"  Task      : {'translate → English' if args.translate else 'transcribe'}")
    print(f"  Stop      : Ctrl+C")
    print(f"{'='*58}\n")

    try:
        import speech_recognition as sr
    except ImportError:
        print(f"{RED}Error: speech_recognition is not installed. Run:{RESET}")
        print(f"  py -m pip install SpeechRecognition PyAudio")
        sys.exit(1)

    # 1. Initialize Microphone
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # 2. Load Whisper model
    print(f"{YELLOW}Loading Whisper '{args.model}' model…{RESET}", flush=True)
    model = whisper.load_model(args.model)
    print(f"{GREEN}Model loaded.{RESET}\n")

    print(f"{YELLOW}Calibrating microphone to ambient noise… please stay quiet for 2 seconds.{RESET}")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
    except Exception as e:
        print(f"\n{RED}{BOLD}CRITICAL MICROPHONE ERROR{RESET}")
        print("Python cannot access your microphone.")
        print("\nFix this in Windows Settings:")
        print("  1. Press Start, type 'Microphone Privacy Settings'")
        print("  2. Turn ON 'Microphone access'")
        print("  3. Turn ON 'Let apps access your microphone'")
        print("  4. Scroll down and turn ON 'Let desktop apps access your microphone'")
        sys.exit(1)
        
    print(f"{GREEN}Ready! Start speaking.{RESET}\n")

    # Print column header
    print(f"{'─'*58}")
    print(f"{'Time':<10}  {'Lang':<8}  Transcription")
    print(f"{'─'*58}")

    task = "translate" if args.translate else "transcribe"
    opts = {"task": task}
    if args.lang:
        opts["language"] = args.lang

    try:
        while True:
            # Record until silence
            with mic as source:
                print(f"{GREY}[Listening…]          {RESET}", end="\r", flush=True)
                audio_data = recognizer.listen(source, phrase_time_limit=10)
                print(f"{GREY}[Transcribing…]       {RESET}", end="\r", flush=True)

            # Save the captured audio chunk to a neat WAV format 
            # (speech_recognition resamples safely internally)
            wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_data)
                wav_path = tmp.name

            try:
                # Transcribe
                result = model.transcribe(wav_path, **opts)
                text = result.get("text", "").strip()
                detected = result.get("language", "??")
            finally:
                os.unlink(wav_path)

            if text:
                ts = time.strftime("%H:%M:%S")
                lang_tag = LANGUAGE_NAMES.get(detected, detected)
                print(f"{GREY}{ts}{RESET}  {CYAN}{lang_tag:<8}{RESET}  {GREEN}{BOLD}{text}{RESET}")
            else:
                print(f"{GREY}[Silent/ignored]        {RESET}", end="\r", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopped.{RESET}")


if __name__ == "__main__":
    main()
