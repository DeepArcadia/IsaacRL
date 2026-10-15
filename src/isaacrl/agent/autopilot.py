"""High-Reliability Autopilot & Dungeon Room Navigator for Isaac."""

import math
import time
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

from isaacrl.core.input import IsaacInputController
from isaacrl.core.vision import DetectedEntity, FrameAnalysis


class IsaacAutopilot:
    """Robust combat and room-to-room navigation engine.
    
    Fixes corner-stuck issues, handles room transitions cleanly, and explores doors.
    """

    def __init__(self, input_controller: IsaacInputController):
        self.inputs = input_controller

        # Navigation state
        self.current_door_idx = 0
        self.door_charge_frames = 0
        self.post_transition_cooldown = 0
        self.last_transition_time = 0.0

        # Anti-Stuck Watchdog
        self.pos_history = deque(maxlen=45)  # ~0.75s of position history at 60 FPS
        self.unstuck_active_frames = 0
        self.unstuck_direction = (0, 0)

        # Potential field parameters for combat
        self.w_tear_repulse = 50000.0
        self.w_enemy_repulse = 35000.0
        self.w_wall_repulse = 18000.0
        self.w_kiting_target = 9.0
        self.kiting_distance = 220.0

    def step(
        self,
        analysis: FrameAnalysis,
        frame_width: int,
        frame_height: int,
        telemetry_transition: bool = False,
    ) -> Tuple[Tuple[int, int], Optional[str]]:
        """Executes one control cycle."""

        # 1. Handle Room Transition Event
        now = time.time()
        if telemetry_transition or (self.post_transition_cooldown > 0):
            if telemetry_transition:
                self.post_transition_cooldown = 40  # Step cleanly into new room for ~0.6s
                self.last_transition_time = now
                # Switch to next door to continue exploring without backtracking
                self.current_door_idx = (self.current_door_idx + 1) % 4
                self.door_charge_frames = 0

            self.post_transition_cooldown -= 1
            # Step slightly towards room center to clear door threshold
            cx, cy = frame_width / 2.0, frame_height / 2.0
            if analysis.isaac:
                dx = 1 if cx > analysis.isaac.x else -1
                dy = 1 if cy > analysis.isaac.y else -1
            else:
                dx, dy = 0, 1
            self.inputs.set_move(dx, dy)
            self.inputs.set_shoot(None)
            return ((dx, dy), None)

        if not analysis.isaac:
            self.inputs.set_move(0, 0)
            self.inputs.set_shoot(None)
            return ((0, 0), None)

        ix, iy = analysis.isaac.x, analysis.isaac.y

        # 2. Check Anti-Stuck Watchdog
        self.pos_history.append((ix, iy))
        if self._check_stuck():
            self.unstuck_active_frames = 30  # Active unstuck for 0.5s
            # Steer 45 degrees toward center
            cx, cy = frame_width / 2.0, frame_height / 2.0
            udx = 1 if cx > ix else -1
            udy = 1 if cy > iy else -1
            self.unstuck_direction = (udx, udy)

        if self.unstuck_active_frames > 0:
            self.unstuck_active_frames -= 1
            dx, dy = self.unstuck_direction
            self.inputs.set_move(dx, dy)
            # If enemies exist, keep firing while unsticking
            shoot_dir = self._compute_shooting_direction(ix, iy, analysis.enemies)
            self.inputs.set_shoot(shoot_dir)
            return ((dx, dy), shoot_dir)

        # 3. Corner Escape (Emergency centering if near room corner)
        if self._is_near_corner(ix, iy, frame_width, frame_height):
            cx, cy = frame_width / 2.0, frame_height / 2.0
            dx = 1 if cx > ix else -1
            dy = 1 if cy > iy else -1
            shoot_dir = self._compute_shooting_direction(ix, iy, analysis.enemies)
            self.inputs.set_move(dx, dy)
            self.inputs.set_shoot(shoot_dir)
            return ((dx, dy), shoot_dir)

        # 4. Mode Selection: COMBAT vs ROOM NAVIGATION
        if not analysis.room_cleared and (analysis.enemies or analysis.enemy_tears):
            # COMBAT MODE
            self.door_charge_frames = 0
            dx, dy = self._compute_combat_movement(ix, iy, analysis, frame_width, frame_height)
            shoot_dir = self._compute_shooting_direction(ix, iy, analysis.enemies)

            self.inputs.set_move(dx, dy)
            self.inputs.set_shoot(shoot_dir)
            return ((dx, dy), shoot_dir)
        else:
            # ROOM CLEARED: DOOR NAVIGATION MODE
            self.inputs.set_shoot(None)
            dx, dy = self._navigate_to_door(ix, iy, analysis.doors, frame_width, frame_height)
            self.inputs.set_move(dx, dy)
            return ((dx, dy), None)

    def _check_stuck(self) -> bool:
        """Returns True if Isaac has stayed in virtually the exact same position for 45 frames."""
        if len(self.pos_history) < 40:
            return False
        first_x, first_y = self.pos_history[0]
        last_x, last_y = self.pos_history[-1]
        dist = math.hypot(last_x - first_x, last_y - first_y)
        return dist < 12.0

    def _is_near_corner(self, ix: float, iy: float, w: int, h: int) -> bool:
        """Checks if Isaac is dangerously close to any of the 4 room corners."""
        margin_x = w * 0.16
        margin_y = h * 0.22
        in_left = ix < margin_x
        in_right = ix > (w - margin_x)
        in_top = iy < margin_y
        in_bot = iy > (h - margin_y)
        return (in_left and in_top) or (in_left and in_bot) or (in_right and in_top) or (in_right and in_bot)

    def _compute_combat_movement(
        self, ix: float, iy: float, analysis: FrameAnalysis, w: int, h: int
    ) -> Tuple[int, int]:
        fx, fy = 0.0, 0.0

        # Repulsion from enemy bullets
        for tear in analysis.enemy_tears:
            dist = math.hypot(ix - tear.x, iy - tear.y) + 1e-4
            if dist < 280.0:
                force = self.w_tear_repulse / (dist * dist)
                fx += ((ix - tear.x) / dist) * force
                fy += ((iy - tear.y) / dist) * force

        # Repulsion & Kiting from enemies
        closest_enemy = None
        min_enemy_dist = float("inf")

        for enemy in analysis.enemies:
            dist = math.hypot(ix - enemy.x, iy - enemy.y) + 1e-4
            if dist < min_enemy_dist:
                min_enemy_dist = dist
                closest_enemy = enemy

            if dist < self.kiting_distance:
                force = self.w_enemy_repulse / (dist * dist)
                fx += ((ix - enemy.x) / dist) * force
                fy += ((iy - enemy.y) / dist) * force

        # Attraction towards kiting sweet spot
        if closest_enemy and min_enemy_dist > (self.kiting_distance + 80.0):
            fx += ((closest_enemy.x - ix) / min_enemy_dist) * self.w_kiting_target
            fy += ((closest_enemy.y - iy) / min_enemy_dist) * self.w_kiting_target

        # Wall repulsion (ONLY in combat mode!)
        left_bound = w * 0.16
        right_bound = w * 0.84
        top_bound = h * 0.22
        bot_bound = h * 0.82

        if ix < left_bound:
            fx += self.w_wall_repulse / (max(ix, 1.0) ** 1.5)
        elif ix > right_bound:
            fx -= self.w_wall_repulse / (max(w - ix, 1.0) ** 1.5)

        if iy < top_bound:
            fy += self.w_wall_repulse / (max(iy, 1.0) ** 1.5)
        elif iy > bot_bound:
            fy -= self.w_wall_repulse / (max(h - iy, 1.0) ** 1.5)

        dx = 0
        dy = 0
        th = 3.5
        if fx > th:
            dx = 1
        elif fx < -th:
            dx = -1

        if fy > th:
            dy = 1
        elif fy < -th:
            dy = -1

        return (dx, dy)

    def _compute_shooting_direction(self, ix: float, iy: float, enemies: list) -> Optional[str]:
        if not enemies:
            return None
        closest = min(enemies, key=lambda e: (e.x - ix) ** 2 + (e.y - iy) ** 2)
        diff_x = closest.x - ix
        diff_y = closest.y - iy
        if abs(diff_x) >= abs(diff_y):
            return "right" if diff_x > 0 else "left"
        else:
            return "down" if diff_y > 0 else "up"

    def _navigate_to_door(
        self, ix: float, iy: float, doors: List[DetectedEntity], w: int, h: int
    ) -> Tuple[int, int]:
        """Directs Isaac straight into an open door to enter the next room.
        
        NOTE: Wall repulsion is COMPLETELY DISABLED during door navigation!
        """
        if not doors:
            return (0, 0)

        # Pick the current target door
        target_door = doors[self.current_door_idx % len(doors)]
        tx, ty = target_door.x, target_door.y

        diff_x = tx - ix
        diff_y = ty - iy
        dist = math.hypot(diff_x, diff_y)

        # If we have reached or are pushing into the door frame for > 4 seconds with no transition, rotate door!
        self.door_charge_frames += 1
        if self.door_charge_frames > 240:  # 4 seconds at 60 FPS
            self.current_door_idx = (self.current_door_idx + 1) % len(doors)
            self.door_charge_frames = 0
            # Jiggle away first
            return (0, 0)

        # Walk directly towards and through the door
        dx = 0
        dy = 0
        deadzone = 12.0

        if abs(diff_x) > deadzone:
            dx = 1 if diff_x > 0 else -1
        if abs(diff_y) > deadzone:
            dy = 1 if diff_y > 0 else -1

        return (dx, dy)
