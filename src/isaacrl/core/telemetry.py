"""Real-time game telemetry stream reader from Isaac's log.txt and Lua bridge."""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple


class IsaacTelemetry:
    """Monitors The Binding of Isaac log.txt in real-time for zero-latency game events."""

    LOG_PATHS = [
        os.path.expanduser("~\\Documents\\My Games\\Binding of Isaac Repentance+\\log.txt"),
        os.path.expanduser("~\\Documents\\My Games\\Binding of Isaac Repentance\\log.txt"),
    ]

    def __init__(self):
        self.log_file: Optional[str] = None
        for p in self.LOG_PATHS:
            if os.path.exists(p):
                self.log_file = p
                break

        self.is_cleared: bool = False
        self.is_game_over: bool = False
        self.current_room_name: str = ""
        self.room_transition_active: bool = False
        self.last_transition_time: float = 0.0

        # Lua bridge ground truth data (if mod enabled)
        self.lua_player_pos: Optional[Tuple[int, int]] = None
        self.lua_hp: int = 6
        self.lua_open_doors: List[Tuple[int, int, int]] = []  # (slot, x, y)

        self._running = True
        self._thread = threading.Thread(target=_tail_log_thread, args=(self,), daemon=True)
        self._thread.start()

    def get_status(self) -> Dict:
        return {
            "room_cleared": self.is_cleared,
            "game_over": self.is_game_over,
            "room": self.current_room_name,
            "transition": self.room_transition_active,
            "open_doors": self.lua_open_doors,
        }

    def close(self):
        self._running = False


def _tail_log_thread(telemetry: IsaacTelemetry):
    if not telemetry.log_file or not os.path.exists(telemetry.log_file):
        return

    try:
        with open(telemetry.log_file, "r", encoding="utf-8", errors="ignore") as f:
            # Seek near end of file
            f.seek(0, os.SEEK_END)

            while telemetry._running:
                line = f.readline()
                if not line:
                    time.sleep(0.01)
                    continue

                line_str = line.strip()

                # 1. Room Transition detection
                if "Starting room transition" in line_str:
                    telemetry.room_transition_active = True
                    telemetry.last_transition_time = time.time()
                    telemetry.is_cleared = False

                elif "Room " in line_str and "(" in line_str:
                    # New room initialized
                    telemetry.current_room_name = line_str
                    telemetry.room_transition_active = False

                # 2. Game Over detection
                elif "Game Over" in line_str:
                    telemetry.is_game_over = True

                elif "Adding collectible" in line_str or "SpawnRNG seed" in line_str:
                    telemetry.is_game_over = False

                # 3. Ground-truth data from IsaacRL Bridge Mod
                elif "[ISAACRL_DATA]" in line_str:
                    # Parse: frame=... clear=true/false x=... y=... hp=... doors=slot:x:y,...
                    try:
                        parts = line_str.split("[ISAACRL_DATA]")[1].strip().split()
                        kv = {}
                        for p in parts:
                            if "=" in p:
                                k, v = p.split("=", 1)
                                kv[k] = v

                        if "clear" in kv:
                            telemetry.is_cleared = (kv["clear"].lower() == "true")
                        if "x" in kv and "y" in kv:
                            telemetry.lua_player_pos = (int(kv["x"]), int(kv["y"]))
                        if "hp" in kv:
                            telemetry.lua_hp = int(kv["hp"])
                        if "doors" in kv and kv["doors"]:
                            doors = []
                            for d in kv["doors"].split(","):
                                if d:
                                    slot, dx, dy = map(int, d.split(":"))
                                    doors.append((slot, dx, dy))
                            telemetry.lua_open_doors = doors
                    except Exception:
                        pass
    except Exception:
        pass
