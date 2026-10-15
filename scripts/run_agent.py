"""Main runner for the IsaacRL Autonomous Agent with Room Navigation & Background Telemetry."""

import argparse
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from isaacrl.agent.autopilot import IsaacAutopilot
from isaacrl.core.capture import ScreenCapture
from isaacrl.core.input import IsaacInputController
from isaacrl.core.telemetry import IsaacTelemetry
from isaacrl.core.vision import IsaacVisionDetector
from isaacrl.core.window import IsaacWindowManager


def main():
    parser = argparse.ArgumentParser(description="IsaacRL Autonomous Gaming Agent")
    parser.add_argument("--mode", choices=["autopilot", "gym"], default="autopilot", help="Agent mode")
    parser.add_argument("--debug", action="store_true", help="Display real-time visualizer window")
    parser.add_argument("--target-fps", type=int, default=60, help="Target loop FPS (default: 60)")
    parser.add_argument("--auto-focus", action="store_true", default=True, help="Automatically focus game window")
    args = parser.parse_args()

    print("=" * 65)
    print(" 🚀 IsaacRL - Autonomous Agent for The Binding of Isaac")
    print(" Organization: DeepArcadia | Full-Run Room Navigator")
    print("=" * 65)

    # 1. Connect to Isaac Window
    window_mgr = IsaacWindowManager()
    if not window_mgr.is_running():
        print("❌ Error: Could not locate running window of 'The Binding of Isaac'.")
        print("Please launch Isaac via Steam and run this script again.")
        sys.exit(1)

    if args.auto_focus:
        window_mgr.bring_to_front()
        time.sleep(0.3)

    rect = window_mgr.get_client_rect()
    print(f"✅ Found Isaac Window: HWND=0x{window_mgr.hwnd:X}, Client Rect={rect}")

    # 2. Initialize Subsystems & Real-time Telemetry
    telemetry = IsaacTelemetry()
    capture = ScreenCapture(window_mgr)
    detector = IsaacVisionDetector()
    inputs = IsaacInputController()
    autopilot = IsaacAutopilot(inputs)

    print("⚡ Real-time Telemetry & Motion Vision Active.")
    print(f"🎯 Target FPS: {args.target_fps} | Ready for Live Play! (Ctrl+C to stop)")
    time.sleep(0.8)

    frame_interval = 1.0 / args.target_fps
    frame_count = 0

    try:
        while True:
            loop_start = time.time()

            # A. Grab frame
            frame = capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # B. Read game telemetry (ground truth room state & transitions)
            t_status = telemetry.get_status()
            telemetry_cleared = t_status["room_cleared"] if t_status["room_cleared"] else None
            transition_active = t_status["transition"]

            # C. Real-time Computer Vision
            analysis = detector.process(
                frame, draw_debug=args.debug, telemetry_cleared=telemetry_cleared
            )

            # D. Compute & Execute Actions via Autopilot
            (dx, dy), shoot_dir = autopilot.step(
                analysis, w, h, telemetry_transition=transition_active
            )

            frame_count += 1
            fps = capture.fps

            # E. Status Logging (every 30 frames)
            if frame_count % 30 == 0:
                isaac_str = (
                    f"({analysis.isaac.x:.0f}, {analysis.isaac.y:.0f})"
                    if analysis.isaac
                    else "Tracking..."
                )
                enemies_cnt = len(analysis.enemies)
                tears_cnt = len(analysis.enemy_tears)
                mode_str = "NAV_DOOR" if analysis.room_cleared else "COMBAT"
                door_target = autopilot.current_door_idx
                print(
                    f"[{mode_str}] Door:{door_target} | FPS:{fps:4.1f} | Isaac:{isaac_str:<12} | "
                    f"Enemies:{enemies_cnt:2d} | Tears:{tears_cnt:2d} | "
                    f"Move:({dx:+d}, {dy:+d}) | Shoot:{str(shoot_dir):<5}"
                )

            # F. Optional Visualizer Window
            if args.debug and analysis.processed_frame is not None:
                vis = analysis.processed_frame
                hud_text = f"IsaacRL | FPS: {fps:.1f} | Mode: {'DOOR NAV' if analysis.room_cleared else 'COMBAT'}"
                cv2.putText(vis, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"Move: ({dx:+d}, {dy:+d}) | Shoot: {shoot_dir} | Target Door: {autopilot.current_door_idx}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("IsaacRL - Realtime HUD (Press Q to exit)", cv2.resize(vis, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Frame rate pacing
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Stopping IsaacRL agent...")
    finally:
        inputs.release_all()
        capture.close()
        telemetry.close()
        if args.debug:
            cv2.destroyAllWindows()
        print("✅ Cleaned up: All keys released, telemetry closed.")


if __name__ == "__main__":
    main()
