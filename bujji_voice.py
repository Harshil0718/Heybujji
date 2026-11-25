import asyncio
from collections import deque
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, cast

import edge_tts
import numpy as np
import pyaudio
import pygame
import requests
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from openwakeword import utils as openwakeword_utils
from openwakeword.model import Model as OpenWakeWordModel

def _sf_load(filepath, *_args, **_kwargs):
    del _args, _kwargs
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _sf_save(filepath, src, sample_rate, *_args, **_kwargs):
    del _args, _kwargs
    arr = src.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr.T
    sf.write(str(filepath), arr, sample_rate)


torchaudio.load = _sf_load
torchaudio.save = _sf_save

sys.path.append(str(Path(__file__).resolve().parent.parent / "mcp_servers"))
from laptop_tools_mcp import (
    laptop_device_control,
    laptop_dictate_message,
    laptop_open_app,
    laptop_set_alarm,
    learn_new_skill,
    recall_learned_skills,
)
from phone_bridge_mcp import (
    phone_device_control,
    phone_dictate_message,
    phone_make_call,
    phone_open_app,
    phone_set_alarm,
)

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "ref_audio" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NVIDIA_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("BUJJI_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_STT_ENABLED = os.getenv("NVIDIA_STT_ENABLED", "false").lower() == "true"
HUB_URL = f"http://127.0.0.1:{os.getenv('HUB_PORT', '8765')}"

WAKEWORD_MODEL_PATH = os.getenv("WAKEWORD_MODEL_PATH", "models/hey_bujji.onnx")
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.25"))
REF_AUDIO_PATH = str((BASE_DIR / os.getenv("REF_AUDIO_PATH", "ref_audio/bujji_voice.wav")).resolve())
REF_AUDIO_TEXT = os.getenv("REF_AUDIO_TEXT", "Honestly brings out the best of people as well.")

F5_NFE_STEPS = 8
TELUGU_MODE = False

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL, timeout=12.0, max_retries=1)

print(f"[1/3] Loading Speech-to-Text ({'NVIDIA Cloud Whisper + Local Fallback' if NVIDIA_STT_ENABLED else 'Local Faster-Whisper'})...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print(f"[2/3] Loading Fast F5-TTS Voice Cloner (nfe_step={F5_NFE_STEPS}) using: {REF_AUDIO_PATH}")
f5_tts_engine = None
if Path(REF_AUDIO_PATH).exists():
    try:
        from f5_tts.api import F5TTS

        f5_tts_engine = F5TTS()
        print("[F5-TTS] Custom reference voice loaded successfully!")
    except Exception as exc:
        print(f"[F5-TTS ERROR] Could not load F5-TTS: {exc}")

print("[3/3] Loading Offline Wake-Word Engine...")
openwakeword_utils.download_models()
if Path(WAKEWORD_MODEL_PATH).exists():
    oww_model = OpenWakeWordModel(wakeword_models=[WAKEWORD_MODEL_PATH], inference_framework="onnx")
else:
    oww_model = OpenWakeWordModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")

pygame.mixer.init()

TOOL_DISPATCH = {
    "laptop_open_app": laptop_open_app,
    "laptop_device_control": laptop_device_control,
    "laptop_dictate_message": laptop_dictate_message,
    "laptop_set_alarm": laptop_set_alarm,
    "phone_open_app": phone_open_app,
    "phone_make_call": phone_make_call,
    "phone_dictate_message": phone_dictate_message,
    "phone_device_control": phone_device_control,
    "phone_set_alarm": phone_set_alarm,
    "learn_new_skill": learn_new_skill,
    "recall_learned_skills": recall_learned_skills,
}

WAKE_PATTERN = re.compile(r"(?i)\b(bujji|buji|buzzy|budgie|boogie|buchi|puji|hey\s+buddy)\b")
STRIP_WAKE_PATTERN = re.compile(r"(?i)^(?:[\s,\.\!\?]*(?:hey\s+)?(?:bujji|buji|buzzy|budgie|boogie|buchi|puji|buddy)[\s,\.\!\?]*)+")
WHISPER_PROMPT = "Hey Bujji, talk Telugu, speak English, cheppandi, decrease volume to 50, open Notepad, Chrome, VS Code."


def _build_system_prompt() -> str:
    lang_rule = (
        "CRITICAL LANGUAGE RULE: You MUST reply in natural, fluent conversational Telugu script (తెలుగు) right now."
        if TELUGU_MODE
        else "LANGUAGE RULE: Reply in fluent, natural Indian English (1 to 2 short sentences)."
    )
    return f"""You are Bujji, a warm, witty, intelligent personal AI voice assistant with a South Indian female personality.
{lang_rule}
Never output markdown formatting, asterisks, or emojis because your reply is read aloud via TTS.

When an action is needed, reply ONLY with a valid JSON object:
{{"tool": "<tool_name>", "args": {{...}}}}

Available tools:
- laptop_open_app: {{"app_name": "Notepad"}}
- laptop_device_control: {{"action": "set_volume|volume_up|volume_down|mute|unmute|set_brightness|brightness_up|brightness_down|media_play_pause|media_next|media_prev|lock_screen", "value": 50}}
- laptop_dictate_message: {{"text": "hello", "app_name": "WhatsApp", "send_message": false}}
- laptop_set_alarm: {{"time_24h": "07:30", "label": "Bujji Alarm"}}
- phone_open_app: {{"app_name": "WhatsApp"}}
- phone_make_call: {{"contact_or_number": "Mom"}}
- phone_dictate_message: {{"text": "On my way", "send": false}}
- phone_device_control: {{"action": "set_volume|mute|unmute|set_brightness|dnd_on|dnd_off", "value": 50}}
- phone_set_alarm: {{"hour": 7, "minute": 30, "label": "Alarm"}}
- learn_new_skill: {{"trigger_phrase": "coding mode", "description": "Opens VS Code and dims volume", "action_plan": "Open VS Code and set volume to 30"}}
- recall_learned_skills: {{"query": "coding mode"}}

If no tool is needed, answer directly in plain spoken text."""


def contains_telugu_script(text: str) -> bool:
    return bool(re.search(r"[\u0C00-\u0C7F]", text))


def _play_audio(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()


async def _synthesize_edge_tts(text: str, out_path: str):
    voice = "te-IN-ShrutiNeural" if contains_telugu_script(text) else "en-IN-NeerjaNeural"
    communicate = edge_tts.Communicate(text, voice=voice, rate="+5%")
    await communicate.save(out_path)


def speak(text: str, cache_name: str | None = None):
    clean_speech = re.sub(r"[*_`#]", "", text).strip()
    print(f"\nBujji: {clean_speech}")

    if f5_tts_engine is not None and not contains_telugu_script(clean_speech):
        if cache_name:
            cached_wav = CACHE_DIR / f"{cache_name}.wav"
            if cached_wav.exists():
                _play_audio(str(cached_wav))
                return
            try:
                f5_tts_engine.infer(
                    ref_file=REF_AUDIO_PATH,
                    ref_text=REF_AUDIO_TEXT,
                    gen_text=clean_speech,
                    file_wave=str(cached_wav),
                    nfe_step=F5_NFE_STEPS,
                    cfg_strength=2,
                    speed=1.0,
                    remove_silence=True,
                )
                _play_audio(str(cached_wav))
                return
            except Exception as exc:
                print(f"[F5-TTS Cache Warning] {exc}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        try:
            f5_tts_engine.infer(
                ref_file=REF_AUDIO_PATH,
                ref_text=REF_AUDIO_TEXT,
                gen_text=clean_speech,
                file_wave=wav_path,
                nfe_step=F5_NFE_STEPS,
                cfg_strength=2,
                speed=1.0,
                remove_silence=True,
            )
            _play_audio(wav_path)
            return
        except Exception as exc:
            print(f"[F5-TTS Runtime Warning] {exc}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        mp3_path = tmp.name
    try:
        asyncio.run(_synthesize_edge_tts(clean_speech, mp3_path))
        _play_audio(mp3_path)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def transcribe_audio(audio_np: np.ndarray) -> str:
    if len(audio_np) < 9600:
        return ""

    if NVIDIA_STT_ENABLED and NVIDIA_API_KEY:
        try:
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())
            buf.seek(0)
            r = requests.post(
                f"{NVIDIA_BASE_URL}/audio/transcriptions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                files={"file": ("speech.wav", buf, "audio/wav")},
                data={"model": "openai/whisper-large-v3", "language": "en"},
                timeout=6,
            )
            if r.status_code == 200:
                txt = r.json().get("text", "").strip()
                if txt:
                    return txt
        except Exception:
            pass

    segments, _ = whisper_model.transcribe(
        audio_np,
        beam_size=2,
        language="en",
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def _pick_live_model() -> str:
    global MODEL_NAME
    try:
        skip = ("embed", "rerank", "guard", "vision", "parse", "ocr", "reward", "safety", "whisper", "tts")
        for m in client.models.list().data:
            if any(w in m.id.lower() for w in skip):
                continue
            try:
                client.chat.completions.create(
                    model=m.id,
                    messages=[{"role": "user", "content": "OK"}],
                    max_tokens=3,
                )
                MODEL_NAME = m.id
                print(f"[Model Auto-Switch] Connected to active model: {MODEL_NAME}")
                return MODEL_NAME
            except Exception:
                continue
    except Exception as e:
        print(f"[Model Discovery Error] {e}")
    return MODEL_NAME


def run_agent_turn(user_text: str, history: list[dict]) -> str:
    global TELUGU_MODE

    if re.search(r"(?i)\b(talk|speak|switch\s+to|reply\s+in)\s+telugu\b|\btelugu\s+lo\b", user_text):
        TELUGU_MODE = True
    elif re.search(r"(?i)\b(talk|speak|switch\s+to|reply\s+in)\s+english\b", user_text):
        TELUGU_MODE = False

    history.append({"role": "user", "content": user_text})

    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=cast(Any, [{"role": "system", "content": _build_system_prompt()}] + history[-10:]),
                temperature=0.25,
                max_tokens=200,
            )
        except Exception as exc:
            if "410" in str(exc) or "404" in str(exc):
                _pick_live_model()
                continue
            print(f"[LLM Error] {exc}")
            return "Sorry, I could not reach the AI server right now."

        content = (resp.choices[0].message.content or "").strip()
        match = re.search(r'\{.*"tool"\s*:\s*"[^"]+".*\}', content, re.DOTALL)
        if match:
            try:
                call_obj = json.loads(match.group(0))
                fn_name = call_obj["tool"]
                fn_args = call_obj.get("args", {})
                print(f"  [Tool] {fn_name}({fn_args})")
                fn = TOOL_DISPATCH.get(fn_name)
                result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"
                history.append({"role": "assistant", "content": content})
                history.append({"role": "user", "content": f"Tool result: {result}. Give your short spoken confirmation."})
                continue
            except Exception:
                pass

        history.append({"role": "assistant", "content": content})
        return content or "Done."

    return "Completed your actions."


def record_audio_array(
    stream: pyaudio.Stream,
    initial_frames: list[bytes] | None = None,
    sample_rate: int = 16000,
    chunk_size: int = 1280,
    silence_threshold: float = 520.0,
    silence_duration: float = 1.0,
    max_duration: float = 10.0,
) -> np.ndarray:
    frames: list[bytes] = list(initial_frames) if initial_frames else []
    silent_chunks = 0
    max_silent_chunks = int(silence_duration * sample_rate / chunk_size)
    max_chunks = int(max_duration * sample_rate / chunk_size)
    speech_started = bool(initial_frames)

    for _ in range(max_chunks):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))

        if rms > silence_threshold:
            speech_started = True
            silent_chunks = 0
        elif speech_started:
            silent_chunks += 1
            if silent_chunks >= max_silent_chunks:
                break

    raw_pcm = b"".join(frames)
    return np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0


def notification_poller():
    while True:
        try:
            r = requests.get(f"{HUB_URL}/internal/notifications", timeout=2)
            if r.status_code == 200:
                for item in r.json().get("notifications", []):
                    speak(f"Important alert from {item['app']}: {item['title']}. {item['text']}")
        except Exception:
            pass
        time.sleep(3)


def main():
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1280
    SPEECH_RMS_TRIGGER = 600.0
    history: list[dict] = []
    pre_roll: deque[bytes] = deque(maxlen=8)
    follow_up_until = 0.0

    threading.Thread(target=notification_poller, daemon=True).start()

    pa = pyaudio.PyAudio()
    default_in = pa.get_default_input_device_info()
    print(f"[Audio] Using Microphone: {default_in['name']}")
    print(f"[AI Engine] Active Model: {MODEL_NAME}")

    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    speak("Bujji online and ready.", cache_name="startup_fast_v1")
    print("\n[Listening! Say 'Hey Bujji <your command>'...]")

    try:
        while True:
            pcm_bytes = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            pre_roll.append(pcm_bytes)
            frame = np.frombuffer(pcm_bytes, dtype=np.int16)
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

            scores = cast(dict[str, float], oww_model.predict(frame))
            oww_triggered = any(score >= WAKEWORD_THRESHOLD for score in scores.values())

            if oww_triggered:
                print("\n[Wake Word Triggered!] Listening for command...")
                oww_model.reset()
                pre_roll.clear()
                speak("Cheppandi?", cache_name="cheppandi_fast_v1")
                audio_np = record_audio_array(mic_stream, None, SAMPLE_RATE, CHUNK_SIZE)
                transcript = transcribe_audio(audio_np)

                if transcript:
                    print(f"You: {transcript}")
                    reply = run_agent_turn(transcript, history)
                    speak(reply)
                    follow_up_until = time.time() + 10.0
                oww_model.reset()
                print("\n[Listening! Say 'Hey Bujji <your command>'...]")

            elif rms > SPEECH_RMS_TRIGGER:
                buffered = list(pre_roll)
                pre_roll.clear()
                audio_np = record_audio_array(mic_stream, buffered, SAMPLE_RATE, CHUNK_SIZE)
                transcript = transcribe_audio(audio_np)

                if transcript:
                    print(f"[Heard] {transcript}")
                    in_follow_up = time.time() < follow_up_until
                    if WAKE_PATTERN.search(transcript) or in_follow_up:
                        cleaned = STRIP_WAKE_PATTERN.sub("", transcript).strip() if WAKE_PATTERN.search(transcript) else transcript
                        if not cleaned:
                            speak("Cheppandi, I am listening.", cache_name="listening_fast_v1")
                            follow_up_until = time.time() + 10.0
                        else:
                            reply = run_agent_turn(cleaned, history)
                            speak(reply)
                            follow_up_until = time.time() + 10.0
                        oww_model.reset()
                        print("\n[Listening! Say 'Hey Bujji <your command>'...]")

    except KeyboardInterrupt:
        print("\nStopping Bujji.")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()