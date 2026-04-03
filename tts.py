"""
tts.py — Unified TTS with true real-time streaming playback.

HOW STREAMING WORKS NOW (vs before):

BEFORE (file-based):
  API → download ALL bytes → write to _tts.mp3 → sf.read() → sd.play()
  User hears nothing until the ENTIRE audio is downloaded.

AFTER (chunk streaming):
  API → receive first chunk (~200ms) → decode → play immediately
       → receive next chunk while previous is playing → ...
  User hears first word in ~300-500ms regardless of answer length.

IMPLEMENTATION:
  OpenAI/Deepgram/ElevenLabs all support HTTP chunked transfer.
  pydub decodes mp3 chunks → raw PCM float32 → sounddevice.OutputStream.
  Decoding runs in a background thread so HTTP receive and audio playback
  run concurrently — no stall between chunks.

INSTALL:
  pip install pydub
  brew install ffmpeg      (Mac)
  apt install ffmpeg       (Linux)

Set TTS_PROVIDER in .env:
  TTS_PROVIDER=openai      (recommended)
  TTS_PROVIDER=deepgram
  TTS_PROVIDER=elevenlabs
"""

import io
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from config import (
    TTS_PROVIDER,
    DEEPGRAM_API_KEY, DEEPGRAM_TTS_MODEL,
    OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
)

log = logging.getLogger(__name__)

PLAYBACK_SAMPLE_RATE = 24000   # OpenAI TTS native rate; Deepgram is 24kHz too
CHANNELS             = 1
HTTP_CHUNK_BYTES     = 4096    # bytes per HTTP read — smaller = lower latency to first audio
MIN_DECODE_BYTES     = 8192    # min bytes to accumulate before pydub decodes
                               # (too small = incomplete mp3 frame = decode error)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: stream mp3 bytes → decode → play in real-time
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_and_play(audio_bytes_iter) -> float:
    """
    Takes any iterator of raw mp3 bytes and plays them in real-time.

    Two concurrent threads:
      decode_worker  — receives HTTP chunks, decodes mp3 → float32 PCM, queues it
      main thread    — drains PCM queue, writes to sounddevice.OutputStream

    This means the speaker starts playing the first decoded chunk while
    the HTTP connection is still downloading the rest.

    Args:
        audio_bytes_iter: Iterator[bytes] — from httpx.iter_bytes()

    Returns:
        Total seconds from function call to playback end.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "Install pydub for streaming TTS:\n"
            "  pip install pydub\n"
            "  brew install ffmpeg    # Mac\n"
            "  apt install ffmpeg     # Linux"
        )

    start         = time.time()
    pcm_queue     = queue.Queue(maxsize=32)
    first_audio_t = [None]   # list so decode_worker can write to it

    def decode_worker():
        buffer = b""
        for raw_chunk in audio_bytes_iter:
            if not raw_chunk:
                continue
            buffer += raw_chunk

            # Accumulate MIN_DECODE_BYTES before attempting decode
            # — pydub needs a complete mp3 frame header to parse correctly
            if len(buffer) >= MIN_DECODE_BYTES:
                try:
                    seg   = AudioSegment.from_mp3(io.BytesIO(buffer))
                    seg   = seg.set_frame_rate(PLAYBACK_SAMPLE_RATE).set_channels(CHANNELS)
                    pcm   = np.array(seg.get_array_of_samples(), dtype=np.int16)
                    f32   = pcm.astype(np.float32) / 32768.0
                    pcm_queue.put(f32)
                    buffer = b""   # reset after successful decode
                except Exception:
                    pass           # incomplete frame — keep accumulating

        # Flush any remaining bytes after the HTTP stream closes
        if buffer:
            try:
                seg   = AudioSegment.from_mp3(io.BytesIO(buffer))
                seg   = seg.set_frame_rate(PLAYBACK_SAMPLE_RATE).set_channels(CHANNELS)
                pcm   = np.array(seg.get_array_of_samples(), dtype=np.int16)
                f32   = pcm.astype(np.float32) / 32768.0
                pcm_queue.put(f32)
            except Exception as e:
                log.warning(f"[TTS] Final flush warning: {e}")

        pcm_queue.put(None)   # sentinel — tells playback loop we're done

    decoder = threading.Thread(target=decode_worker, daemon=True)
    decoder.start()

    # Playback loop — runs in main thread while decoder runs concurrently
    with sd.OutputStream(
        samplerate=PLAYBACK_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    ) as stream:
        while True:
            pcm_chunk = pcm_queue.get()
            if pcm_chunk is None:
                break
            if first_audio_t[0] is None:
                first_audio_t[0] = time.time()
                log.info(f"[TTS] ⚡ First audio in {first_audio_t[0] - start:.3f}s")
            stream.write(pcm_chunk)

    decoder.join()
    total = time.time() - start
    log.info(f"[TTS] Done — total={total:.2f}s")
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI TTS
# ═══════════════════════════════════════════════════════════════════════════════

def _speak_openai(text: str) -> float:
    """
    Streams OpenAI TTS mp3 directly to speaker — no file on disk.

    Before: collect all bytes → save _tts_oai.mp3 → sf.read → sd.play
    After:  HTTP chunk arrives → decode → speaker starts immediately
    """
    import httpx

    start = time.time()
    try:
        with httpx.stream(
            "POST",
            "https://api.openai.com/v1/audio/speech",
            json={
                "model":           OPENAI_TTS_MODEL,
                "voice":           OPENAI_TTS_VOICE,
                "input":           text,
                "response_format": "mp3",
            },
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=30,
        ) as r:
            r.raise_for_status()
            return _stream_and_play(r.iter_bytes(chunk_size=HTTP_CHUNK_BYTES))
    except Exception as e:
        log.error(f"[OpenAI TTS] {e}")
        print(f"[TTS fallback — not spoken] {text}")
        return time.time() - start


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPGRAM TTS
# ═══════════════════════════════════════════════════════════════════════════════

def _speak_deepgram(text: str) -> float:
    """
    Streams Deepgram TTS mp3 directly to speaker.
    Uses raw httpx — no SpeakOptions import, works with any deepgram SDK version.
    """
    import httpx

    start = time.time()
    try:
        with httpx.stream(
            "POST",
            f"https://api.deepgram.com/v1/speak?model={DEEPGRAM_TTS_MODEL}&encoding=mp3",
            json={"text": text},
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=30,
        ) as r:
            r.raise_for_status()
            return _stream_and_play(r.iter_bytes(chunk_size=HTTP_CHUNK_BYTES))
    except Exception as e:
        log.error(f"[Deepgram TTS] {e}")
        print(f"[TTS fallback — not spoken] {text}")
        return time.time() - start


# ═══════════════════════════════════════════════════════════════════════════════
# ELEVENLABS TTS
# ═══════════════════════════════════════════════════════════════════════════════

def _speak_elevenlabs(text: str) -> float:
    """Streams ElevenLabs TTS mp3 directly to speaker."""
    import httpx

    start = time.time()
    try:
        with httpx.stream(
            "POST",
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
            json={
                "text":           text,
                "model_id":       ELEVENLABS_MODEL_ID,
                "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
            },
            headers={
                "xi-api-key":   ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=30,
        ) as r:
            r.raise_for_status()
            return _stream_and_play(r.iter_bytes(chunk_size=HTTP_CHUNK_BYTES))
    except Exception as e:
        log.error(f"[ElevenLabs TTS] {e}")
        print(f"[TTS fallback — not spoken] {text}")
        return time.time() - start


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC
# ═══════════════════════════════════════════════════════════════════════════════

def speak(text: str) -> float:
    """Speak text using TTS_PROVIDER. Returns total seconds (API + playback)."""
    provider = TTS_PROVIDER.lower()
    log.info(f"[TTS] Provider: {provider}")
    print("🔊 Speaking...")

    if provider == "openai":
        return _speak_openai(text)
    elif provider == "deepgram":
        return _speak_deepgram(text)
    elif provider == "elevenlabs":
        return _speak_elevenlabs(text)
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {provider!r}. Choose: openai | deepgram | elevenlabs")