import os
import requests
from dotenv import load_dotenv
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("bujji_phone")
HUB_URL = f"http://127.0.0.1:{os.getenv('HUB_PORT', '8765')}"


def _dispatch_to_phone(action: str, params: dict) -> str:
    try:
        resp = requests.post(
            f"{HUB_URL}/internal/send_to_phone",
            json={"action": action, "params": params},
            timeout=12,
        )
        data = resp.json()
        if resp.status_code == 200:
            return data.get("result", "Executed on phone.")
        return f"Phone not connected yet ({data.get('detail', 'offline')}). Connect your mobile client to ws://<LAPTOP_IP>:8765/ws/phone."
    except Exception as exc:
        return f"Bidirectional Hub unreachable: {exc}"


@mcp.tool()
def phone_open_app(app_name: str) -> str:
    """Open any app on the user's phone by name."""
    return _dispatch_to_phone("open_app", {"app_name": app_name})


@mcp.tool()
def phone_make_call(contact_or_number: str) -> str:
    """Make a voice call on the user's phone."""
    return _dispatch_to_phone("make_call", {"target": contact_or_number})


@mcp.tool()
def phone_dictate_message(text: str, send: bool = False) -> str:
    """Type speech-to-text dictated text into the currently open messaging app on the phone."""
    return _dispatch_to_phone("inject_text", {"text": text, "send": send})


@mcp.tool()
def phone_device_control(action: str, value: int = 50) -> str:
    """Control phone volume, mute, brightness, or DND."""
    return _dispatch_to_phone("device_control", {"action": action, "value": value})


@mcp.tool()
def phone_set_alarm(hour: int, minute: int, label: str = "Bujji Alarm") -> str:
    """Set a native alarm on the phone (24-hour clock)."""
    return _dispatch_to_phone("set_alarm", {"hour": hour, "minute": minute, "label": label})


if __name__ == "__main__":
    mcp.run()
