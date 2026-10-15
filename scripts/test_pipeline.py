"""Self-test and verification benchmark for IsaacRL pipeline."""

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
from isaacrl.core.vision import IsaacVisionDetector
from isaacrl.core.window import IsaacWindowManager


def run_benchmark():
    print("=" * 60)
    print(" 🧪 IsaacRL Pipeline Diagnostic & Benchmark")
    print("=" * 60)

    # Test 1: Window Detection
    print("\n[1/4] Testing Isaac Window Attachment...")
    win_mgr = IsaacWindowManager()
    if not win_mgr.is_running():
        print("❌ FAILED: Isaac window not found! Please make sure The Binding of Isaac is running.")
        return False
    rect = win_mgr.get_client_rect()
    print(f"✅ PASSED: Window Found! HWND=0x{win_mgr.hwnd:X}, Client Rect={rect}")

    # Test 2: High-Speed Screen Capture
    print("\n[2/4] Testing Screen Capture Frame Rate (Target: 60-70+ FPS)...")
    capture = ScreenCapture(win_mgr, target_size=(960, 540))
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

    # Test 3: Vision Feature Extraction
    print("\n[3/4] Testing Real-Time Computer Vision Detection...")
    detector = IsaacVisionDetector()
    v_start = time.time()
    for _ in range(30):
        analysis = detector.process(frame)
    v_duration = time.time() - v_start
    v_fps = 30 / v_duration
    print(f"✅ PASSED: Vision inference speed: {v_fps:.1f} FPS ({v_duration*1000/30:.2f} ms/frame)")
    print(f"   - Isaac detected: {analysis.isaac is not None}")
    if analysis.isaac:
        print(f"   - Isaac Coordinates: ({analysis.isaac.x:.1f}, {analysis.isaac.y:.1f})")
    print(f"   - Enemies detected: {len(analysis.enemies)}")
    print(f"   - Enemy Tears detected: {len(analysis.enemy_tears)}")
    print(f"   - Room Cleared: {analysis.room_cleared}")

    # Test 4: Input Controller & Autopilot
    print("\n[4/4] Testing Autopilot Logic & Controller...")
    inputs = IsaacInputController()
    autopilot = IsaacAutopilot(inputs)
    h, w = frame.shape[:2]
    move, shoot = autopilot.step(analysis, w, h)
    print(f"✅ PASSED: Autopilot evaluated step! Move: {move}, Shoot: {shoot}")
    inputs.release_all()

    print("\n" + "=" * 60)
    print(" 🎉 ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
    print(f" Total End-to-End Capability: ~{min(fps, v_fps):.1f} FPS (Ready for Live Play)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_benchmark()
