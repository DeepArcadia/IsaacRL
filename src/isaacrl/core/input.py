"""Low-latency DirectInput keyboard simulation for Isaac."""

import ctypes
import time
from typing import Set


# Win32 SendInput Structures
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1


class IsaacInputController:
    """Simulates DirectInput keyboard events with hardware scancodes."""

    # Hardware DirectInput Scan Codes
    SCANCODES = {
        # Movement
        "w": (0x11, False),
        "a": (0x1E, False),
        "s": (0x1F, False),
        "d": (0x20, False),
        # Shooting
        "up": (0x48, True),
        "down": (0x50, True),
        "left": (0x4B, True),
        "right": (0x4D, True),
        # Actions
        "space": (0x39, False),
        "e": (0x12, False),
        "q": (0x10, False),
        "shift": (0x2A, False),
        "r": (0x13, False),
        "tab": (0x0F, False),
        "esc": (0x01, False),
    }

    def __init__(self):
        self._currently_pressed: Set[str] = set()

    def press_key(self, key_name: str) -> None:
        """Presses and holds a key down."""
        key_name = key_name.lower()
        if key_name not in self.SCANCODES or key_name in self._currently_pressed:
            return

        code, extended = self.SCANCODES[key_name]
        flags = KEYEVENTF_SCANCODE
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, code, flags, 0, ctypes.pointer(extra))
        x = Input(INPUT_KEYBOARD, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._currently_pressed.add(key_name)

    def release_key(self, key_name: str) -> None:
        """Releases a currently held key."""
        key_name = key_name.lower()
        if key_name not in self.SCANCODES or key_name not in self._currently_pressed:
            return

        code, extended = self.SCANCODES[key_name]
        flags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, code, flags, 0, ctypes.pointer(extra))
        x = Input(INPUT_KEYBOARD, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._currently_pressed.discard(key_name)

    def tap_key(self, key_name: str, duration: float = 0.05) -> None:
        """Taps a key for a short duration."""
        self.press_key(key_name)
        time.sleep(duration)
        self.release_key(key_name)

    def release_all(self) -> None:
        """Releases all currently held keys."""
        for key in list(self._currently_pressed):
            self.release_key(key)

    def set_move(self, dx: int, dy: int) -> None:
        """Sets movement keys based on dx (-1, 0, 1) and dy (-1, 0, 1)."""
        # Horizontal
        if dx < 0:
            self.press_key("a")
            self.release_key("d")
        elif dx > 0:
            self.press_key("d")
            self.release_key("a")
        else:
            self.release_key("a")
            self.release_key("d")

        # Vertical (dy > 0 means down in screen coordinates)
        if dy < 0:
            self.press_key("w")
            self.release_key("s")
        elif dy > 0:
            self.press_key("s")
            self.release_key("w")
        else:
            self.release_key("w")
            self.release_key("s")

    def set_shoot(self, shoot_dir: str) -> None:
        """Sets shooting keys: 'up', 'down', 'left', 'right', or None/'none'."""
        shoot_keys = ["up", "down", "left", "right"]
        if not shoot_dir or shoot_dir.lower() == "none":
            for k in shoot_keys:
                self.release_key(k)
            return

        target_key = shoot_dir.lower()
        for k in shoot_keys:
            if k == target_key:
                self.press_key(k)
            else:
                self.release_key(k)

    def restart_run(self, hold_seconds: float = 1.2) -> None:
        """Holds 'R' to restart the current Isaac run."""
        self.release_all()
        self.press_key("r")
        time.sleep(hold_seconds)
        self.release_key("r")
        time.sleep(0.5)

    def __del__(self):
        self.release_all()
