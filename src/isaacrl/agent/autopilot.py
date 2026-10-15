"""Reactive Potential-Field Combat Autopilot & Navigator for Isaac."""

import math
from typing import Optional, Tuple
import numpy as np
from isaacrl.core.input import IsaacInputController
from isaacrl.core.vision import DetectedEntity, FrameAnalysis


class IsaacAutopilot:
    """Zero-shot reactive combat engine using Potential-Field physics for dodging & aiming."""

    def __init__(self, input_controller: IsaacInputController):
        self.inputs = input_controller

        # Potential field weight hyperparameters
        self.w_tear_repulse = 45000.0
        self.w_enemy_repulse = 30000.0
        self.w_wall_repulse = 15000.0
        self.w_kiting_target = 8.0

        # Kiting sweet spot (distance in pixels to maintain from closest enemy)
        self.kiting_distance = 220.0

        # Door exploration state
        self.current_door_target: Optional[DetectedEntity] = None
        self.door_switch_cooldown = 0

    def step(self, analysis: FrameAnalysis, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Optional[str]]:
        """Computes the movement (dx, dy) and shoot direction ('up','down','left','right',None)
        given the latest frame analysis.
        """
        if not analysis.isaac:
            # If Isaac is not currently detected, release controls to prevent drifting
            self.inputs.set_move(0, 0)
            self.inputs.set_shoot(None)
            return ((0, 0), None)

        ix, iy = analysis.isaac.x, analysis.isaac.y

        # Mode A: Combat State (Enemies or Projectiles present)
        if analysis.enemies or analysis.enemy_tears:
            self.current_door_target = None
            dx, dy = self._compute_combat_movement(ix, iy, analysis, frame_width, frame_height)
            shoot_dir = self._compute_shooting_direction(ix, iy, analysis.enemies)

            self.inputs.set_move(dx, dy)
            self.inputs.set_shoot(shoot_dir)
            return ((dx, dy), shoot_dir)

        # Mode B: Exploration / Room Cleared State (Navigate to Doors)
        else:
            self.inputs.set_shoot(None)
            dx, dy = self._compute_door_navigation(ix, iy, analysis.doors, frame_width, frame_height)
            self.inputs.set_move(dx, dy)
            return ((dx, dy), None)

    def _compute_combat_movement(
        self, ix: float, iy: float, analysis: FrameAnalysis, w: int, h: int
    ) -> Tuple[int, int]:
        """Calculates repulsive & attractive potential vector field forces."""
        fx, fy = 0.0, 0.0

        # 1. Strong Repulsion from Enemy Tears (Bullets)
        for tear in analysis.enemy_tears:
            dist = math.hypot(ix - tear.x, iy - tear.y) + 1e-4
            if dist < 280.0:
                # Inverse-square repulsion
                force = self.w_tear_repulse / (dist * dist)
                fx += ((ix - tear.x) / dist) * force
                fy += ((iy - tear.y) / dist) * force

        # 2. Repulsion / Kiting from Enemies
        closest_enemy = None
        min_enemy_dist = float("inf")

        for enemy in analysis.enemies:
            dist = math.hypot(ix - enemy.x, iy - enemy.y) + 1e-4
            if dist < min_enemy_dist:
                min_enemy_dist = dist
                closest_enemy = enemy

            if dist < self.kiting_distance:
                # Too close: push away
                force = self.w_enemy_repulse / (dist * dist)
                fx += ((ix - enemy.x) / dist) * force
                fy += ((iy - enemy.y) / dist) * force

        # 3. Kiting Attraction (Stay within range if too far away)
        if closest_enemy and min_enemy_dist > (self.kiting_distance + 100.0):
            fx += ((closest_enemy.x - ix) / min_enemy_dist) * self.w_kiting_target
            fy += ((closest_enemy.y - iy) / min_enemy_dist) * self.w_kiting_target

        # 4. Wall Repulsion (Keep away from room boundaries and spikes)
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

        # Quantize net force to discrete movement
        dx = 0
        dy = 0
        threshold = 3.5

        if fx > threshold:
            dx = 1
        elif fx < -threshold:
            dx = -1

        if fy > threshold:
            dy = 1
        elif fy < -threshold:
            dy = -1

        return (dx, dy)

    def _compute_shooting_direction(self, ix: float, iy: float, enemies: list) -> Optional[str]:
        """Chooses the primary axis to shoot at the closest high-threat enemy."""
        if not enemies:
            return None

        # Find closest enemy
        closest = min(enemies, key=lambda e: (e.x - ix) ** 2 + (e.y - iy) ** 2)
        diff_x = closest.x - ix
        diff_y = closest.y - iy

        # Compare horizontal vs vertical displacement
        if abs(diff_x) >= abs(diff_y):
            return "right" if diff_x > 0 else "left"
        else:
            return "down" if diff_y > 0 else "up"

    def _compute_door_navigation(
        self, ix: float, iy: float, doors: list, w: int, h: int
    ) -> Tuple[int, int]:
        """Moves towards doors to explore the next room when room is cleared."""
        if not doors:
            return (0, 0)

        # Select a door target (default to top/right or rotate)
        if not self.current_door_target or self.door_switch_cooldown <= 0:
            # Prefer top or right door first
            self.current_door_target = doors[0]  # top door
            self.door_switch_cooldown = 180
        else:
            self.door_switch_cooldown -= 1

        tx, ty = self.current_door_target.x, self.current_door_target.y
        diff_x = tx - ix
        diff_y = ty - iy

        dx = 0
        dy = 0
        deadzone = 20.0

        if abs(diff_x) > deadzone:
            dx = 1 if diff_x > 0 else -1
        if abs(diff_y) > deadzone:
            dy = 1 if diff_y > 0 else -1

        return (dx, dy)
