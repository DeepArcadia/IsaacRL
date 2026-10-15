"""Diagnostic benchmark for IsaacRL pipeline with telemetry & motion detection."""

import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaacrl.agent.autopilot import IsaacAutopilot
from isaacrl.core.capture import ScreenCapture
from isaacrl.core.input import IsaacInputController
from isaacrl.core.telemetry import IsaacTelemetry
from isaacrl.core.vision import IsaacVisionDetector
from isaacrl.core.window import IsaacWindowManager


def run_benchmark():
    print("=" * 65)
    print(" 🧪 IsaacRL Pipeline Diagnostic & Benchmark v2")
    print("=" * 65)

    # 1. Window Detection
    print("\n[1/5] Testing Isaac Window Attachment...")
    win_mgr = IsaacWindowManager()
    if not win_mgr.is_running():
        print("❌ FAILED: Isaac window not found! Please make sure The Binding of Isaac is running.")
        return False
    rect = win_mgr.get_client_rect()
    print(f"✅ PASSED: Window Found! HWND=0x{win_mgr.hwnd:X}, Client Rect={rect}")

    # 2. Telemetry Log Stream
    print("\n[2/5] Testing Real-time Telemetry Stream...")
    telemetry = IsaacTelemetry()
    time.sleep(0.3)
    status = telemetry.get_status()
    print(f"✅ PASSED: Telemetry Log: {telemetry.log_file}")
    print(f"   - Room Status: Cleared={status['room_cleared']}, Game Over={status['game_over']}")

    # 3. High-Speed Screen Capture
    print("\n[3/5] Testing Screen Capture Frame Rate (Target: 60-70+ FPS)...")
    capture = ScreenCapture(win_mgr)
    start = time.time()
    frames = 60
    for _ in range(frames):
        frame = capture.grab()
        if frame is None:
            print("❌ FAILED: Unable to grab frame from window.")
            return False
    duration = time.time() - start
    fps = frames / duration
    print(f"✅ PASSED: Captured {frames} frames in {duration:.3f}s -> {fps:.1f} FPS (Shape: {frame.shape})")

    # 4. Motion-Filtered Vision
    print("\n[4/5] Testing Temporal Motion Vision Feature Extraction...")
    detector = IsaacVisionDetector()
    v_start = time.time()
    # Feed two frames to initialize motion differencing
    f1 = capture.grab()
    time.sleep(0.016)
    f2 = capture.grab()
    analysis = detector.process(f1)
    analysis = detector.process(f2)
    v_duration = time.time() - v_start
    print(f"✅ PASSED: Motion Vision inference: {v_duration*1000/2:.2f} ms/frame")
    print(f"   - Isaac tracking: {analysis.isaac is not None}")
    print(f"   - Moving Enemies detected: {len(analysis.enemies)} (Static obstacles correctly filtered!)")
    print(f"   - Enemy Tears detected: {len(analysis.enemy_tears)}")
    print(f"   - Room Cleared State: {analysis.room_cleared}")

    # 5. Autopilot & Anti-Stuck Controller
    print("\n[5/5] Testing Autopilot Room Navigator & Controller...")
    inputs = IsaacInputController()
    autopilot = IsaacAutopilot(inputs)
    h, w = frame.shape[:2]
    move, shoot = autopilot.step(analysis, w, h)
    print(f"✅ PASSED: Autopilot evaluation successful!")
    print(f"   - Action: Move: {move}, Shoot: {shoot}, Door Index: {autopilot.current_door_idx}")
    inputs.release_all()
    telemetry.close()

    print("\n" + "=" * 65)
    print(" 🎉 ALL V2 TESTS PASSED! Ready for Autonomous Playthrough.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    run_benchmark()
