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
    Uses Deepgram's streaming/live transcription via WebSocket.
    Sends the recorded audio as a single stream burst — still gives you
    the streaming path (word-level timestamps, endpointing) vs batch.
    """
    import asyncio
    from deepgram import (
        DeepgramClient, LiveTranscriptionEvents,
        LiveOptions, DeepgramClientOptions,
    )

    transcript_parts = []
    done_event = asyncio.Event()

    async def _run():
        config = DeepgramClientOptions(options={"keepalive": "true"})
        dg = DeepgramClient(api_key=DEEPGRAM_API_KEY, config=config)
        conn = dg.listen.asynclive.v("1")

        def on_message(self, result, **kwargs):
            alt = result.channel.alternatives[0]
            if alt.transcript and result.is_final:
                transcript_parts.append(alt.transcript)

        def on_close(self, close, **kwargs):
            done_event.set()

        def on_error(self, error, **kwargs):
            log.error(f"[Deepgram streaming] {error}")
            done_event.set()

        conn.on(LiveTranscriptionEvents.Transcript, on_message)
        conn.on(LiveTranscriptionEvents.Close,      on_close)
        conn.on(LiveTranscriptionEvents.Error,      on_error)

        opts = LiveOptions(
            model=DEEPGRAM_STT_MODEL,
            language="en",
            smart_format=True,
            punctuate=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
        )
        await conn.start(opts)

        # Send audio in 100ms chunks (simulates real streaming)
        chunk_size = int(SAMPLE_RATE * 0.1)
        pcm = (audio * 32767).astype(np.int16).tobytes()
        for i in range(0, len(pcm), chunk_size * 2):  # *2 because int16=2 bytes
            await conn.send(pcm[i : i + chunk_size * 2])
            await asyncio.sleep(0.01)

        await conn.finish()
        # Wait for close event (max 3s)
        try:
            await asyncio.wait_for(done_event.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

    start = time.time()
    try:
        asyncio.run(_run())
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