"""Window management and desktop attachment for The Binding of Isaac."""

from typing import Optional, Tuple
import win32gui
import win32service
import win32con
import win32process


class IsaacWindowManager:
    """Manages finding, desktop attachment, and geometry of the Isaac game window."""

    TITLE_KEYWORDS = [
        "Binding of Isaac: Repentance+",
        "Binding of Isaac: Repentance",
        "The Binding of Isaac: Rebirth",
        "Binding of Isaac",
    ]

    def __init__(self):
        self._ensure_default_desktop()
        self.hwnd: Optional[int] = None
        self.refresh()

    def _ensure_default_desktop(self) -> None:
        """Ensures the current thread is attached to the interactive Default desktop."""
        try:
            desk = win32service.OpenDesktop("Default", 0, False, win32con.GENERIC_ALL)
            desk.SetThreadDesktop()
        except Exception:
            pass

    def find_window(self) -> Optional[int]:
        """Finds the HWND of the running Isaac window."""
        self._ensure_default_desktop()

        found_hwnd = None

        def enum_cb(hwnd, _):
            nonlocal found_hwnd
            if found_hwnd is None and win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                for kw in self.TITLE_KEYWORDS:
                    if kw in title:
                        found_hwnd = hwnd
                        break
            return True

        try:
            win32gui.EnumWindows(enum_cb, None)
        except Exception:
            pass
        return found_hwnd

    def refresh(self) -> bool:
        """Refreshes the window handle."""
        self.hwnd = self.find_window()
        return self.hwnd is not None

    def is_running(self) -> bool:
        """Returns True if the Isaac process and window are currently active."""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return self.refresh()
        return True

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns (left, top, right, bottom) of the game window."""
        if not self.is_running():
            return None
        return win32gui.GetWindowRect(self.hwnd)

    def get_client_rect(self, force_refresh: bool = False) -> Optional[Tuple[int, int, int, int]]:
        """Returns (left, top, width, height) of the client area in screen coordinates."""
        if not self.hwnd or force_refresh:
            if not self.refresh():
                return None
        try:
            point = win32gui.ClientToScreen(self.hwnd, (0, 0))
            _, _, w, h = win32gui.GetClientRect(self.hwnd)
            return (point[0], point[1], w, h)
        except Exception:
            self.refresh()
            return None

    def bring_to_front(self) -> bool:
        """Brings the game window to the foreground."""
        if not self.is_running():
            return False
        try:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
            return True
        except Exception:
            return False
