import io
import time
import wave
from pathlib import Path
import numpy as np
import requests
from openai import OpenAI

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=4.0, max_retries=0)

print("[1/3] Fetching active models from NVIDIA NIM...")
all_models = [m.id for m in client.models.list().data]

skip = ("embed", "rerank", "guard", "vision", "parse", "ocr", "reward", "safety", "bge-", "nvclip", "deplot", "kosmos", "whisper", "tts")
candidates = [m for m in all_models if not any(s in m.lower() for s in skip)]

priority_keywords = ("llama-4", "llama-3.3", "llama-3.1", "qwen2.5", "mistral-small", "mistral-large", "gemma-2", "gemma-3")
prioritized = [m for m in candidates if any(k in m.lower() for k in priority_keywords)]
others = [m for m in candidates if m not in prioritized]
test_list = (prioritized + others)[:15]

print("[2/3] Testing response latency (4s timeout per model)...")
fastest_model = None
best_latency = 999.0

for model_id in test_list:
    print(f"  -> Probing {model_id}...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=3,
            temperature=0.1,
        )
        dt = time.perf_counter() - t0
        if resp.choices and resp.choices[0].message.content:
            print(f"OK ({dt:.2f}s)")
            if dt < best_latency:
                best_latency = dt
                fastest_model = model_id
            if dt < 1.5:
                break
        else:
            print("empty")
    except Exception:
        print("skip")

if not fastest_model:
    raise RuntimeError("No active chat model responded within 4s. Check your internet connection.")

print(f"\n[SELECTED FASTEST MODEL] {fastest_model} ({best_latency:.2f}s)")

print("[3/3] Checking NVIDIA NIM Cloud Whisper STT...")
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
buf.seek(0)

nvidia_stt_ok = False
try:
    r = requests.post(
        f"{BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"file": ("test.wav", buf, "audio/wav")},
        data={"model": "openai/whisper-large-v3"},
        timeout=4,
    )
    nvidia_stt_ok = r.status_code == 200
except Exception:
    nvidia_stt_ok = False

print(f"[NVIDIA CLOUD STT] {'ENABLED' if nvidia_stt_ok else 'Using Fast Local Faster-Whisper'}")

env_text = f"""OPENAI_API_KEY={API_KEY}
OPENAI_BASE_URL={BASE_URL}
BUJJI_MODEL={fastest_model}
NVIDIA_STT_ENABLED={'true' if nvidia_stt_ok else 'false'}

OPENJARVIS_URL=http://127.0.0.1:8000/v1

HUB_HOST=0.0.0.0
HUB_PORT=8765
HUB_SECRET=bujji-secret-2026

REF_AUDIO_PATH=ref_audio/bujji_voice.wav
REF_AUDIO_TEXT=Honestly brings out the best of people as well.
WAKEWORD_MODEL_PATH=models/hey_bujji.onnx
WAKEWORD_THRESHOLD=0.25
"""
Path(".env").write_text(env_text, encoding="utf-8")

toml_text = f"""[engine]
default = "openai"
base_url = "{BASE_URL}"
api_key_env = "OPENAI_API_KEY"

[intelligence]
default_model = "{fastest_model}"
temperature = 0.2

[agent]
default_agent = "orchestrator"
memory_enabled = true
learning_enabled = true
skills_dir = "C:/bujji-pc/learned_skills"
system_prompt = \"\"\"
You are Bujji, a warm, witty, intelligent personal AI voice assistant with a South Indian female personality.
Rules:
1. When asked to speak Telugu, reply naturally in Telugu script.
2. Otherwise, reply in crisp, natural Indian English (1 to 2 sentences).
3. Use `laptop_*` tools for PC/laptop actions and `phone_*` tools for mobile phone actions.
\"\"\"

[server]
host = "127.0.0.1"
port = 8000

[mcp_servers.bujji_laptop]
command = "C:/bujji-pc/.venv/Scripts/python.exe"
args = ["C:/bujji-pc/mcp_servers/laptop_tools_mcp.py"]

[mcp_servers.bujji_phone]
command = "C:/bujji-pc/.venv/Scripts/python.exe"
args = ["C:/bujji-pc/mcp_servers/phone_bridge_mcp.py"]
"""
Path("configs/config.toml").write_text(toml_text, encoding="utf-8")
user_cfg = Path.home() / ".openjarvis" / "config.toml"
user_cfg.parent.mkdir(parents=True, exist_ok=True)
user_cfg.write_text(toml_text, encoding="utf-8")
print("[SAVED] .env, configs/config.toml, and ~/.openjarvis/config.toml")