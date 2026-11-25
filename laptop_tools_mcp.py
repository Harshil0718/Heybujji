import json
import os
import subprocess
import threading
import time
import winreg
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, cast

import comtypes
import pyautogui
import pyperclip
import screen_brightness_control as sbc

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from fastmcp import FastMCP
from pycaw.pycaw import AudioUtilities

mcp = FastMCP("bujji_laptop")

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = BASE_DIR / "learned_skills"
SKILLS_DB = SKILLS_DIR / "skills_registry.json"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
if not SKILLS_DB.exists():
    SKILLS_DB.write_text("[]", encoding="utf-8")


def _get_volume_interface() -> Any:
    comtypes.CoInitialize()
    speakers = cast(Any, AudioUtilities.GetSpeakers())
    return speakers.EndpointVolume


def _scan_installed_apps() -> dict[str, str]:
    apps: dict[str, str] = {}
    start_dirs = [
        Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / r"Microsoft\Windows\Start Menu\Programs",
        Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
    ]
    for root in start_dirs:
        if root.exists():
            for lnk in root.rglob("*.lnk"):
                apps[lnk.stem.lower()] = str(lnk)

    reg_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
    ]
    for hive, subkey in reg_paths:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    exe_name = winreg.EnumKey(key, i)
                    with winreg.OpenKey(key, exe_name) as app_key:
                        path_val, _ = winreg.QueryValueEx(app_key, "")
                        clean_name = exe_name.lower().replace(".exe", "")
                        apps[clean_name] = path_val.strip('"')
        except OSError:
            continue
    return apps


@mcp.tool()
def laptop_open_app(app_name: str) -> str:
    """Open any application installed on the Windows laptop/PC by name."""
    target = app_name.strip().lower()
    apps = _scan_installed_apps()
    for name, path in apps.items():
        if target == name or target in name:
            os.startfile(path)
            return f"Opened {name} on your laptop."

    ps_cmd = (
        f"$app = Get-StartApps | Where-Object {{ $_.Name -like '*{app_name}*' }} | Select-Object -First 1; "
        f"if ($app) {{ Start-Process ('shell:AppsFolder\\' + $app.AppID); Write-Output $app.Name }}"
    )
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
    if res.stdout.strip():
        return f"Launched {res.stdout.strip()} on your laptop."
    return f"Could not find an application matching '{app_name}' on your laptop."


@mcp.tool()
def laptop_device_control(action: str, value: Optional[int] = None) -> str:
    """Control Windows laptop volume, brightness, media playback, or lock screen."""
    act = action.lower().strip()
    if act in {"set_volume", "volume_up", "volume_down", "mute", "unmute"}:
        vol = _get_volume_interface()
        current = round(vol.GetMasterVolumeLevelScalar() * 100)
        if act == "set_volume" and value is not None:
            clamped = max(0, min(100, int(value)))
            vol.SetMasterVolumeLevelScalar(clamped / 100.0, None)
            return f"Laptop volume set to {clamped}%."
        if act == "volume_up":
            new_v = min(100, current + 15)
            vol.SetMasterVolumeLevelScalar(new_v / 100.0, None)
            return f"Laptop volume increased to {new_v}%."
        if act == "volume_down":
            new_v = max(0, current - 15)
            vol.SetMasterVolumeLevelScalar(new_v / 100.0, None)
            return f"Laptop volume decreased to {new_v}%."
        if act == "mute":
            vol.SetMute(1, None)
            return "Laptop audio muted."
        if act == "unmute":
            vol.SetMute(0, None)
            return "Laptop audio unmuted."

    if act in {"set_brightness", "brightness_up", "brightness_down"}:
        try:
            curr_b = sbc.get_brightness(display=0)[0]
            if act == "set_brightness" and value is not None:
                clamped = max(0, min(100, int(value)))
                sbc.set_brightness(clamped, display=0)
                return f"Laptop brightness set to {clamped}%."
            if act == "brightness_up":
                new_b = min(100, curr_b + 15)
                sbc.set_brightness(new_b, display=0)
                return f"Laptop brightness increased to {new_b}%."
            if act == "brightness_down":
                new_b = max(0, curr_b - 15)
                sbc.set_brightness(new_b, display=0)
                return f"Laptop brightness decreased to {new_b}%."
        except Exception as exc:
            return f"Brightness adjustment failed: {exc}"

    if act == "media_play_pause":
        pyautogui.press("playpause")
        return "Toggled media playback on laptop."
    if act == "media_next":
        pyautogui.press("nexttrack")
        return "Skipped to next track."
    if act == "media_prev":
        pyautogui.press("prevtrack")
        return "Went to previous track."
    if act == "lock_screen":
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
        return "Locked the laptop screen."
    return f"Unsupported laptop action: {action}"


@mcp.tool()
def laptop_dictate_message(text: str, app_name: str = "", send_message: bool = False) -> str:
    """Insert dictated text into the active messaging window on the laptop."""
    if app_name:
        laptop_open_app(app_name)
        time.sleep(1.2)
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    if send_message:
        time.sleep(0.2)
        pyautogui.press("enter")
        return f"Typed and sent message: '{text}'"
    return f"Typed dictated message: '{text}'"


@mcp.tool()
def laptop_set_alarm(time_24h: str, label: str = "Bujji Alarm") -> str:
    """Set an alarm on the laptop in HH:MM 24-hour format."""
    now = datetime.now()
    hour, minute = map(int, time_24h.strip().split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    delay_sec = (target - now).total_seconds()

    def _alarm_worker():
        time.sleep(delay_sec)
        for _ in range(5):
            print("\a", end="", flush=True)
            time.sleep(0.5)
        ps_popup = (
            f"Add-Type -AssemblyName PresentationFramework; "
            f"[System.Windows.MessageBox]::Show('{label} ({time_24h})', 'Bujji Alarm')"
        )
        subprocess.Popen(["powershell", "-NoProfile", "-Command", ps_popup])

    threading.Thread(target=_alarm_worker, daemon=True).start()
    return f"Alarm '{label}' set for {time_24h} on your laptop."


@mcp.tool()
def learn_new_skill(trigger_phrase: str, description: str, action_plan: str) -> str:
    """Save a newly taught command or multi-step routine for future reuse."""
    skills = json.loads(SKILLS_DB.read_text(encoding="utf-8"))
    slug = "".join(c if c.isalnum() else "_" for c in trigger_phrase.lower()).strip("_")
    entry = {
        "slug": slug,
        "trigger_phrase": trigger_phrase,
        "description": description,
        "action_plan": action_plan,
        "created_at": datetime.now().isoformat(),
    }
    skills = [s for s in skills if s["slug"] != slug]
    skills.append(entry)
    SKILLS_DB.write_text(json.dumps(skills, indent=2), encoding="utf-8")

    skill_folder = SKILLS_DIR / slug
    skill_folder.mkdir(exist_ok=True)
    skill_md = f"---\nname: {slug}\ndescription: {description}\ntrigger: {trigger_phrase}\n---\n\n# {trigger_phrase}\n\n{action_plan}\n"
    (skill_folder / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return f"Learned new skill '{trigger_phrase}'."


@mcp.tool()
def recall_learned_skills(query: str) -> str:
    """Retrieve previously learned custom user routines matching the query."""
    skills = json.loads(SKILLS_DB.read_text(encoding="utf-8"))
    if not skills:
        return "No custom skills learned yet."
    q_words = set(query.lower().split())
    scored = []
    for s in skills:
        haystack = f"{s['trigger_phrase']} {s['description']} {s['action_plan']}".lower()
        overlap = sum(1 for w in q_words if w in haystack)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for score, s in scored[:3] if score > 0] or [s for _, s in scored[:2]]
    return json.dumps(top, indent=2)


if __name__ == "__main__":
    mcp.run()
