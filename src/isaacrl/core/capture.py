"""High-Performance Screen Capture Pipeline (60-70+ FPS) for Isaac."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np
import mss
from isaacrl.core.window import IsaacWindowManager


class ScreenCapture:
    """Ultra-low-latency screen grabber targeting 60-70+ FPS game frame rate."""

    def __init__(self, window_mgr: Optional[IsaacWindowManager] = None, target_size: Optional[Tuple[int, int]] = None):
        self.window_mgr = window_mgr or IsaacWindowManager()
        self.target_size = target_size  # (width, height) or None for native
        self._sct = mss.mss()
        self._last_frame_time = time.time()
        self._fps_history = []

    def grab(self) -> Optional[np.ndarray]:
        """Captures the current frame of the Isaac window as a BGR/RGB numpy array.
        
        Returns:
            np.ndarray of shape (H, W, 3) in BGR format, or None if window is unavailable.
        """
        rect = self.window_mgr.get_client_rect()
        if not rect:
            return None

        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return None

        bbox = {"left": left, "top": top, "width": width, "height": height}

        try:
            raw = self._sct.grab(bbox)
            # Ultra-fast zero-copy reshape from raw byte buffer (500+ FPS)
            frame = np.frombuffer(raw.raw, dtype=np.uint8).reshape((raw.height, raw.width, 4))[:, :, :3]

            if self.target_size and (raw.width, raw.height) != self.target_size:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)

            # Measure frame timing
            now = time.time()
            dt = now - self._last_frame_time
            self._last_frame_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                self._fps_history.append(current_fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)

            return frame
        except Exception:
            return None

    @property
    def fps(self) -> float:
        """Returns the rolling average FPS over the last 30 frames."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)

    def close(self):
        """Releases the mss instance."""
        if hasattr(self, "_sct") and self._sct:
            self._sct.close()

    def __del__(self):
        self.close()
