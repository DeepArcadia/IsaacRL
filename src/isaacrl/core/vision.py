"""High-speed real-time computer vision detector for The Binding of Isaac."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectedEntity:
    x: float
    y: float
    radius: float
    category: str  # 'isaac', 'enemy', 'enemy_tear', 'pickup', 'door', 'pedestal'
    color: Tuple[int, int, int] = (255, 255, 255)
    confidence: float = 1.0


@dataclass
class FrameAnalysis:
    isaac: Optional[DetectedEntity] = None
    enemies: List[DetectedEntity] = field(default_factory=list)
    enemy_tears: List[DetectedEntity] = field(default_factory=list)
    pickups: List[DetectedEntity] = field(default_factory=list)
    doors: List[DetectedEntity] = field(default_factory=list)
    pedestals: List[DetectedEntity] = field(default_factory=list)
    room_cleared: bool = False
    fps: float = 0.0
    processed_frame: Optional[np.ndarray] = None


class IsaacVisionDetector:
    """Fast, low-latency OpenCV feature extractor for Isaac gameplay."""

    def __init__(self):
        # Tracking history for temporal smoothing
        self.prev_isaac_pos: Optional[Tuple[float, float]] = None

    def process(self, frame: np.ndarray, draw_debug: bool = False) -> FrameAnalysis:
        """Processes a single BGR gameplay frame and returns structured entity detections."""
        h, w = frame.shape[:2]
        analysis = FrameAnalysis()

        # Gameplay area mask (exclude top HUD row where hearts/minimap live)
        hud_height = int(h * 0.16)
        play_area = frame[hud_height:, :]
        play_h, play_w = play_area.shape[:2]

        hsv = cv2.cvtColor(play_area, cv2.COLOR_BGR2HSV)

        # 1. Detect Red Enemy Tears (bright saturated red bullets)
        # In HSV: Red wraps around 0/180
        mask_r1 = cv2.inRange(hsv, np.array([0, 160, 160]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 160, 160]), np.array([180, 255, 255]))
        mask_red_tears = cv2.bitwise_or(mask_r1, mask_r2)

        contours_tears, _ = cv2.findContours(mask_red_tears, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # 2. Detect Isaac (Player character)
        # Isaac has light peach/pink head with dark eyes
        # If previous position is known, prioritize candidate closest to previous pos
        mask_isaac = cv2.inRange(hsv, np.array([8, 40, 150]), np.array([24, 160, 255]))
        contours_isaac, _ = cv2.findContours(mask_isaac, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_isaac = None
        min_dist = float("inf")
        room_center = (play_w / 2.0, play_h / 2.0 + hud_height)

        for cnt in contours_isaac:
            area = cv2.contourArea(cnt)
            if 150 < area < 2500:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                global_y = cy + hud_height
                # Distance to expected position
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
            # Fallback to last known position with decaying confidence
            analysis.isaac = DetectedEntity(
                x=self.prev_isaac_pos[0],
                y=self.prev_isaac_pos[1],
                radius=16.0,
                category="isaac",
                color=(0, 200, 200),
                confidence=0.5,
            )

        # 3. Detect Enemies (High-contrast dynamic entities in the room)
        # Using adaptive thresholding & contour filtering in play area
        gray = cv2.cvtColor(play_area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)

        contours_all, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        isaac_pos = (analysis.isaac.x, analysis.isaac.y) if analysis.isaac else None

        for cnt in contours_all:
            area = cv2.contourArea(cnt)
            if 180 < area < 7000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                global_y = cy + hud_height

                # Reject if this contour overlaps Isaac or a wall boundary
                if isaac_pos:
                    dist_to_isaac = np.hypot(cx - isaac_pos[0], global_y - isaac_pos[1])
                    if dist_to_isaac < 35.0:
                        continue

                # Filter out outer room perimeter walls
                margin_x = w * 0.12
                margin_y = h * 0.18
                if cx < margin_x or cx > (w - margin_x) or global_y < margin_y or global_y > (h - margin_y * 0.8):
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

        # 4. Room cleared status
        analysis.room_cleared = (len(analysis.enemies) == 0 and len(analysis.enemy_tears) == 0)

        # 5. Detect Standard Door Waypoints
        door_margin_x = w * 0.08
        door_margin_y = h * 0.14
        mid_x = w / 2.0
        mid_y = (h + hud_height) / 2.0

        analysis.doors = [
            DetectedEntity(x=mid_x, y=door_margin_y + hud_height, radius=24.0, category="door_top"),
            DetectedEntity(x=mid_x, y=h - door_margin_y, radius=24.0, category="door_bottom"),
            DetectedEntity(x=door_margin_x, y=mid_y, radius=24.0, category="door_left"),
            DetectedEntity(x=w - door_margin_x, y=mid_y, radius=24.0, category="door_right"),
        ]

        if draw_debug:
            vis = frame.copy()
            # Draw Isaac
            if analysis.isaac:
                ix, iy = int(analysis.isaac.x), int(analysis.isaac.y)
                ir = int(analysis.isaac.radius)
                cv2.circle(vis, (ix, iy), ir, (0, 255, 255), 2)
                cv2.putText(vis, "ISAAC", (ix - 20, iy - ir - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw Enemies
            for e in analysis.enemies:
                ex, ey = int(e.x), int(e.y)
                er = int(e.radius)
                cv2.circle(vis, (ex, ey), er, (0, 0, 255), 2)
                cv2.putText(vis, "ENEMY", (ex - 20, ey - er - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw Tears
            for t in analysis.enemy_tears:
                tx, ty = int(t.x), int(t.y)
                tr = int(t.radius)
                cv2.circle(vis, (tx, ty), tr, (0, 140, 255), -1)

            # Draw Doors when cleared
            if analysis.room_cleared:
                for d in analysis.doors:
                    dx, dy = int(d.x), int(d.y)
                    cv2.circle(vis, (dx, dy), 16, (0, 255, 0), 2)
                cv2.putText(vis, "ROOM CLEARED", (int(mid_x - 70), int(hud_height + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            analysis.processed_frame = vis

        return analysis
