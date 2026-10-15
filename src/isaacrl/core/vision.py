"""Robust real-time computer vision with motion filtering for IsaacRL."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectedEntity:
    x: float
    y: float
    radius: float
    category: str  # 'isaac', 'enemy', 'enemy_tear', 'door', 'pickup'
    color: Tuple[int, int, int] = (255, 255, 255)
    confidence: float = 1.0


@dataclass
class FrameAnalysis:
    isaac: Optional[DetectedEntity] = None
    enemies: List[DetectedEntity] = field(default_factory=list)
    enemy_tears: List[DetectedEntity] = field(default_factory=list)
    doors: List[DetectedEntity] = field(default_factory=list)
    room_cleared: bool = False
    processed_frame: Optional[np.ndarray] = None


class IsaacVisionDetector:
    """Robust OpenCV detector combining color segmentation with temporal motion filtering.
    
    Prevents stationary obstacles (rocks, statues, poop, floor cracks) from being
    falsely identified as enemies.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_isaac_pos: Optional[Tuple[float, float]] = None
        self.static_frame_counter: int = 0
        self.history_enemies_counts: List[int] = []

    def process(self, frame: np.ndarray, draw_debug: bool = False, telemetry_cleared: Optional[bool] = None) -> FrameAnalysis:
        h, w = frame.shape[:2]
        analysis = FrameAnalysis()

        hud_height = int(h * 0.16)
        play_area = frame[hud_height:, :]
        play_h, play_w = play_area.shape[:2]

        gray = cv2.cvtColor(play_area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        hsv = cv2.cvtColor(play_area, cv2.COLOR_BGR2HSV)

        # 1. Temporal Motion Detection (Eliminate stationary rocks & obstacles)
        motion_mask = np.zeros_like(gray)
        if self.prev_gray is not None:
            diff = cv2.absdiff(blurred, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            # Dilate to connect broken motion regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        self.prev_gray = blurred

        # 2. Detect Red Enemy Tears (Bullets)
        mask_r1 = cv2.inRange(hsv, np.array([0, 160, 160]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 160, 160]), np.array([180, 255, 255]))
        mask_tears = cv2.bitwise_or(mask_r1, mask_r2)

        contours_tears, _ = cv2.findContours(mask_tears, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_tears:
            area = cv2.contourArea(cnt)
            if 15 < area < 800:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                analysis.enemy_tears.append(
                    DetectedEntity(
                        x=cx,
                        y=cy + hud_height,
                        radius=max(radius, 6.0),
                        category="enemy_tear",
                        color=(0, 0, 255),
                    )
                )

        # 3. Detect Isaac (Player character)
        mask_isaac = cv2.inRange(hsv, np.array([6, 35, 140]), np.array([26, 170, 255]))
        contours_isaac, _ = cv2.findContours(mask_isaac, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_isaac = None
        min_dist = float("inf")
        room_center = (play_w / 2.0, play_h / 2.0 + hud_height)

        for cnt in contours_isaac:
            area = cv2.contourArea(cnt)
            if 120 < area < 3000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                global_y = cy + hud_height
                ref_pos = self.prev_isaac_pos if self.prev_isaac_pos else room_center
                dist = (cx - ref_pos[0]) ** 2 + (global_y - ref_pos[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_isaac = DetectedEntity(
                        x=cx,
                        y=global_y,
                        radius=max(radius, 16.0),
                        category="isaac",
                        color=(0, 255, 255),
                    )

        if best_isaac:
            analysis.isaac = best_isaac
            self.prev_isaac_pos = (best_isaac.x, best_isaac.y)
        elif self.prev_isaac_pos:
            analysis.isaac = DetectedEntity(
                x=self.prev_isaac_pos[0],
                y=self.prev_isaac_pos[1],
                radius=16.0,
                category="isaac",
                color=(0, 200, 200),
                confidence=0.5,
            )

        # 4. Detect True Moving Enemies
        # MUST have active motion and NOT overlap Isaac or outer room walls
        isaac_x = analysis.isaac.x if analysis.isaac else play_w / 2.0
        isaac_y = analysis.isaac.y if analysis.isaac else play_h / 2.0 + hud_height

        # Exclude Isaac's position from motion mask
        motion_no_isaac = motion_mask.copy()
        local_iy = int(isaac_y - hud_height)
        cv2.circle(motion_no_isaac, (int(isaac_x), local_iy), 40, 0, -1)

        # Mask out room perimeter wall edges
        margin_x = int(w * 0.12)
        margin_y = int(h * 0.15)
        wall_mask = np.zeros_like(gray)
        wall_mask[margin_y : play_h - margin_y, margin_x : play_w - margin_x] = 255
        motion_filtered = cv2.bitwise_and(motion_no_isaac, wall_mask)

        contours_enemies, _ = cv2.findContours(motion_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_enemies:
            area = cv2.contourArea(cnt)
            if 80 < area < 8000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                global_y = cy + hud_height

                dist_to_isaac = np.hypot(cx - isaac_x, global_y - isaac_y)
                if dist_to_isaac < 35.0:
                    continue

                analysis.enemies.append(
                    DetectedEntity(
                        x=cx,
                        y=global_y,
                        radius=max(radius, 14.0),
                        category="enemy",
                        color=(0, 0, 220),
                    )
                )

        # 5. Room Cleared Resolution
        # Use telemetry if available, otherwise check moving enemies & tears
        if telemetry_cleared is not None:
            analysis.room_cleared = telemetry_cleared
        else:
            # If no moving enemies for several frames, room is cleared
            current_enemy_cnt = len(analysis.enemies)
            self.history_enemies_counts.append(current_enemy_cnt)
            if len(self.history_enemies_counts) > 20:
                self.history_enemies_counts.pop(0)

            # Room cleared when moving enemies and bullets are 0
            analysis.room_cleared = (sum(self.history_enemies_counts) == 0 and len(analysis.enemy_tears) == 0)

        # 6. Door Waypoints (Exact Center coordinates on each wall)
        # In Isaac 16:9 / 4:3 rooms:
        mid_x = w / 2.0
        door_top_y = h * 0.20
        door_bot_y = h * 0.86
        door_left_x = w * 0.10
        door_right_x = w * 0.90
        mid_y = (door_top_y + door_bot_y) / 2.0

        analysis.doors = [
            DetectedEntity(x=mid_x, y=door_top_y, radius=30.0, category="door_top"),
            DetectedEntity(x=door_right_x, y=mid_y, radius=30.0, category="door_right"),
            DetectedEntity(x=mid_x, y=door_bot_y, radius=30.0, category="door_bottom"),
            DetectedEntity(x=door_left_x, y=mid_y, radius=30.0, category="door_left"),
        ]

        if draw_debug:
            vis = frame.copy()
            if analysis.isaac:
                ix, iy = int(analysis.isaac.x), int(analysis.isaac.y)
                ir = int(analysis.isaac.radius)
                cv2.circle(vis, (ix, iy), ir, (0, 255, 255), 2)
                cv2.putText(vis, "ISAAC", (ix - 20, iy - ir - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            for e in analysis.enemies:
                ex, ey = int(e.x), int(e.y)
                er = int(e.radius)
                cv2.circle(vis, (ex, ey), er, (0, 0, 255), 2)

            for t in analysis.enemy_tears:
                tx, ty = int(t.x), int(t.y)
                cv2.circle(vis, (tx, ty), int(t.radius), (0, 140, 255), -1)

            if analysis.room_cleared:
                for d in analysis.doors:
                    dx, dy = int(d.x), int(d.y)
                    cv2.circle(vis, (dx, dy), 16, (0, 255, 0), 2)
                cv2.putText(vis, "ROOM CLEARED -> ENTERING DOOR", (int(w * 0.3), int(hud_height + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            analysis.processed_frame = vis

        return analysis
