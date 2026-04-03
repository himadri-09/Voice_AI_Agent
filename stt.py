"""
stt.py — Unified STT: Deepgram streaming | OpenAI Whisper | ElevenLabs STT

Set STT_PROVIDER in .env:
  STT_PROVIDER=deepgram    (default, streaming)
  STT_PROVIDER=openai      (Whisper via file upload)
  STT_PROVIDER=elevenlabs  (ElevenLabs STT)
"""

import io
import logging
import time
from typing import Tuple

import numpy as np
import soundfile as sf

from config import (
    STT_PROVIDER,
    DEEPGRAM_API_KEY, DEEPGRAM_STT_MODEL,
    OPENAI_API_KEY, OPENAI_STT_MODEL,
    ELEVENLABS_API_KEY,
)

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000


# ═══════════════════════════════════════════════════════════
# DEEPGRAM — streaming (replaces byte-collect approach)
# ═══════════════════════════════════════════════════════════

def _transcribe_deepgram_streaming(audio: np.ndarray) -> Tuple[str, float]:
    """
    Uses Deepgram's streaming/live transcription via WebSocket (v6.x SDK).
    Sends the recorded audio as a single stream burst.
    """
    from deepgram import DeepgramClient

    start = time.time()
    transcript_parts = []

    try:
        dg = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        # Convert audio to PCM bytes (linear16)
        pcm = (audio * 32767).astype(np.int16).tobytes()

        # Use the synchronous context manager API (v6.x)
        with dg.listen.v1.connect(
            model=DEEPGRAM_STT_MODEL,
            language="en",
            smart_format=True,
            punctuate=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
        ) as socket_client:
            # Send all audio at once and finish
            socket_client.send(pcm)
            socket_client.finish()

            # Collect results from the streaming response
            for message in socket_client.listen():
                try:
                    if hasattr(message, 'channel') and message.channel:
                        if hasattr(message.channel, 'alternatives') and message.channel.alternatives:
                            alt = message.channel.alternatives[0]
                            if hasattr(alt, 'transcript') and alt.transcript:
                                if hasattr(message, 'is_final') and message.is_final:
                                    transcript_parts.append(alt.transcript)
                except (AttributeError, IndexError, TypeError) as e:
                    log.debug(f"[Deepgram streaming] Parse error: {e}")

    except ImportError as e:
        log.error(f"[Deepgram import] {e} — Install with: pip install deepgram-sdk")
        return "", time.time() - start
    except Exception as e:
        log.error(f"[Deepgram streaming] Failed: {e}")
        return "", time.time() - start

    transcript = " ".join(transcript_parts).strip()
    return transcript, time.time() - start


# ═══════════════════════════════════════════════════════════
# OPENAI WHISPER
# ═══════════════════════════════════════════════════════════

def _transcribe_openai(audio: np.ndarray) -> Tuple[str, float]:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    buf.name = "audio.wav"  # Whisper needs a filename hint

    start = time.time()
    try:
        resp = client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=buf,
            language="en",
        )
        return resp.text.strip(), time.time() - start
    except Exception as e:
        log.error(f"[OpenAI STT] {e}")
        return "", time.time() - start


# ═══════════════════════════════════════════════════════════
# ELEVENLABS STT
# ═══════════════════════════════════════════════════════════

def _transcribe_elevenlabs(audio: np.ndarray) -> Tuple[str, float]:
    import httpx

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)

    start = time.time()
    try:
        resp = httpx.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"audio": ("audio.wav", buf, "audio/wav")},
            data={"model_id": "scribe_v1"},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        return text, time.time() - start
    except Exception as e:
        log.error(f"[ElevenLabs STT] {e}")
        return "", time.time() - start


# ═══════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═══════════════════════════════════════════════════════════

def transcribe(audio: np.ndarray) -> Tuple[str, float]:
    """
    Transcribe audio using the provider set in STT_PROVIDER env var.
    Returns (transcript, latency_sec).
    """
    provider = STT_PROVIDER.lower()
    log.info(f"[STT] Provider: {provider}")

    if provider == "deepgram":
        transcript, latency = _transcribe_deepgram_streaming(audio)
    elif provider == "openai":
        transcript, latency = _transcribe_openai(audio)
    elif provider == "elevenlabs":
        transcript, latency = _transcribe_elevenlabs(audio)
    else:
        raise ValueError(f"Unknown STT_PROVIDER: {provider!r}. Choose: deepgram | openai | elevenlabs")

    print(f"📝 [{provider.upper()}] You said: {transcript!r}  ({latency:.2f}s)")
    return transcript, latency