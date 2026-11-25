import asyncio
import os
import sys
import uuid
from collections import deque
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from openai import OpenAI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "mcp_servers"))
from laptop_tools_mcp import (
    laptop_device_control,
    laptop_dictate_message,
    laptop_open_app,
    laptop_set_alarm,
)

load_dotenv()

HUB_SECRET = os.getenv("HUB_SECRET", "bujji-secret-2026")
OPENJARVIS_URL = os.getenv("OPENJARVIS_URL", "http://localhost:8000/v1")
NVIDIA_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("BUJJI_MODEL", "meta/llama-3.3-70b-instruct")

app = FastAPI(title="Bujji Bidirectional Laptop <-> Phone Hub")

connected_phone: WebSocket | None = None
pending_phone_requests: dict[str, asyncio.Future] = {}
urgent_notifications: deque[dict] = deque(maxlen=50)

IMPORTANT_KEYWORDS = {"otp", "bank", "debited", "credited", "urgent", "emergency", "delivery", "call", "meeting"}

DIRECT_LAPTOP_TOOLS = {
    "laptop_open_app": laptop_open_app,
    "laptop_device_control": laptop_device_control,
    "laptop_dictate_message": laptop_dictate_message,
    "laptop_set_alarm": laptop_set_alarm,
}


class OutboundPhoneCommand(BaseModel):
    action: str
    params: dict = {}


class SimulatedPhoneInput(BaseModel):
    text: str


def _ask_openjarvis_sync(command_text: str) -> str:
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=OPENJARVIS_URL)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are Bujji running on the user's laptop. Execute the requested laptop action immediately using your tools and reply concisely.",
            },
            {"role": "user", "content": command_text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or "Done on laptop."


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "phone_connected": connected_phone is not None,
        "mobile_ws_endpoint": "/ws/phone?token=<HUB_SECRET>",
    }


@app.websocket("/ws/phone")
async def mobile_websocket_endpoint(ws: WebSocket):
    """
    Open Mobile Integration Endpoint:
    Connect phone client to ws://<LAPTOP_IP>:8765/ws/phone?token=bujji-secret-2026
    """
    global connected_phone
    token = ws.query_params.get("token", "")
    if token != HUB_SECRET:
        await ws.close(code=4001)
        return

    await ws.accept()
    connected_phone = ws
    print("[Bujji Hub] Mobile device connected via WebSocket.")

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            # 1. Phone -> Laptop: Natural language voice/text command via OpenJarvis
            if msg_type == "voice_command":
                user_cmd = msg.get("text", "").strip()
                if user_cmd:
                    print(f"[Phone -> Laptop] {user_cmd}")
                    loop = asyncio.get_running_loop()
                    reply = await loop.run_in_executor(None, _ask_openjarvis_sync, user_cmd)
                    await ws.send_json({"type": "assistant_reply", "text": reply})

            # 2. Phone -> Laptop: Instant direct control (sliders/buttons on phone UI)
            elif msg_type == "direct_tool":
                tool_name = msg.get("tool", "")
                args = msg.get("args", {})
                fn = DIRECT_LAPTOP_TOOLS.get(tool_name)
                result = fn(**args) if fn else f"Unknown laptop tool: {tool_name}"
                await ws.send_json({"type": "assistant_reply", "text": str(result)})

            # 3. Phone -> Laptop: Important notification alert forwarding
            elif msg_type == "notification":
                app_name = msg.get("app", "Phone")
                title = msg.get("title", "")
                text = msg.get("text", "")
                combined = f"{title} {text}".lower()
                if any(k in combined for k in IMPORTANT_KEYWORDS) or "whatsapp" in app_name.lower():
                    urgent_notifications.append({"app": app_name, "title": title, "text": text})

            # 4. Phone -> Laptop: Result of a command sent from Laptop -> Phone
            elif msg_type == "command_result":
                req_id = msg.get("request_id")
                if req_id in pending_phone_requests and not pending_phone_requests[req_id].done():
                    pending_phone_requests[req_id].set_result(msg.get("result", "Completed on phone."))

    except WebSocketDisconnect:
        print("[Bujji Hub] Mobile device disconnected.")
        connected_phone = None


@app.post("/internal/send_to_phone")
async def send_command_to_phone(cmd: OutboundPhoneCommand):
    if connected_phone is None:
        raise HTTPException(status_code=503, detail="No mobile device connected to /ws/phone.")

    req_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending_phone_requests[req_id] = fut

    await connected_phone.send_json({
        "type": "phone_command",
        "request_id": req_id,
        "action": cmd.action,
        "params": cmd.params,
    })

    try:
        result = await asyncio.wait_for(fut, timeout=10.0)
        return {"status": "ok", "result": result}
    except asyncio.TimeoutError:
        return {"status": "timeout", "result": "Sent to phone, awaiting response."}
    finally:
        pending_phone_requests.pop(req_id, None)


@app.get("/internal/notifications")
def pop_urgent_notifications():
    items = list(urgent_notifications)
    urgent_notifications.clear()
    return {"notifications": items}


@app.post("/internal/simulate_phone")
async def simulate_phone_command(payload: SimulatedPhoneInput):
    loop = asyncio.get_running_loop()
    reply = await loop.run_in_executor(None, _ask_openjarvis_sync, payload.text)
    return {"simulated_phone_sent": payload.text, "laptop_executed_and_replied": reply}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HUB_HOST", "0.0.0.0"),
        port=int(os.getenv("HUB_PORT", "8765")),
    )
    