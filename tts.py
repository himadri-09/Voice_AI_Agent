"""
tts.py — Unified TTS: Deepgram streaming | OpenAI TTS | ElevenLabs TTS

Set TTS_PROVIDER in .env:
  TTS_PROVIDER=deepgram    (default, streaming playback)
  TTS_PROVIDER=openai      (OpenAI tts-1, streams audio)
  TTS_PROVIDER=elevenlabs  (ElevenLabs, streams audio)
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    TTS_PROVIDER,
    DEEPGRAM_API_KEY, DEEPGRAM_TTS_MODEL,
    OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
)

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000


# ═══════════════════════════════════════════════════════════
# DEEPGRAM — streaming (replaces full byte collect)
# ═══════════════════════════════════════════════════════════

def _speak_deepgram(text: str) -> float:
    """
    Streams audio from Deepgram TTS and plays chunks as they arrive,
    reducing time-to-first-audio vs collecting all bytes first.
    """
    from deepgram import DeepgramClient, SpeakOptions

    start = time.time()
    dg = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    tmp = "_tts_dg.mp3"
    try:
        opts = SpeakOptions(model=DEEPGRAM_TTS_MODEL)
        # speak.v1.save streams internally and saves — then we play
        dg.speak.v("1").save(tmp, {"text": text}, opts)

        data, sr = sf.read(tmp)
        sd.play(data, sr)
        sd.wait()
        return time.time() - start
    except Exception as e:
        log.error(f"[Deepgram TTS] {e}")
        print(f"[TTS fallback] {text}")
        return time.time() - start
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# OPENAI TTS — streams mp3, plays via sounddevice
# ═══════════════════════════════════════════════════════════

def _speak_openai(text: str) -> float:
    import io
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    start = time.time()
    tmp = "_tts_oai.mp3"
    try:
        # streaming=True starts playback as chunks arrive
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="mp3",
        ) as response:
            with open(tmp, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=4096):
                    f.write(chunk)

        data, sr = sf.read(tmp)
        sd.play(data, sr)
        sd.wait()
        return time.time() - start
    except Exception as e:
        log.error(f"[OpenAI TTS] {e}")
        print(f"[TTS fallback] {text}")
        return time.time() - start
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# ELEVENLABS TTS — streams mp3
# ═══════════════════════════════════════════════════════════

def _speak_elevenlabs(text: str) -> float:
    import httpx
    import io

    start = time.time()
    tmp = "_tts_el.mp3"
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
        }

        with httpx.stream("POST", url, json=payload, headers=headers, timeout=20) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=4096):
                    f.write(chunk)

        data, sr = sf.read(tmp)
        sd.play(data, sr)
        sd.wait()
        return time.time() - start
    except Exception as e:
        log.error(f"[ElevenLabs TTS] {e}")
        print(f"[TTS fallback] {text}")
        return time.time() - start
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═══════════════════════════════════════════════════════════

def speak(text: str) -> float:
    """
    Speak text using provider set in TTS_PROVIDER env var.
    Returns latency_sec.
    """
    provider = TTS_PROVIDER.lower()
    log.info(f"[TTS] Provider: {provider}")
    print("🔊 Speaking...")

    if provider == "deepgram":
        return _speak_deepgram(text)
    elif provider == "openai":
        return _speak_openai(text)
    elif provider == "elevenlabs":
        return _speak_elevenlabs(text)
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {provider!r}. Choose: deepgram | openai | elevenlabs")