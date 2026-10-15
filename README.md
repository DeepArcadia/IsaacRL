# IsaacRL (DeepArcadia)

> High-Performance Reinforcement Learning & Autopilot AI Agent for *The Binding of Isaac: Repentance+*

[![DeepArcadia](https://img.shields.io/badge/Organization-DeepArcadia-blue.svg)](https://github.com/DeepArcadia)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

`IsaacRL` is an autonomous gaming agent and reinforcement learning platform designed specifically for *The Binding of Isaac: Repentance+*. 

It combines **ultra-low-latency 60+ FPS screen capture**, **real-time computer vision**, a **Potential-Field Reactive Combat Autopilot** for dodging/kiting, and a full **Gymnasium Environment** for end-to-end Deep RL training.

### Key Features

- **⚡ 60-70+ FPS High-Frequency Pipeline**: Direct memory screen grab (`mss`) and vectorized NumPy operations matching the game's internal 60 tick rate.
- **🛡️ Reactive Potential-Field Combat Engine**:
  - Continuous tear and enemy avoidance via inverse-square repulsive vector fields.
  - Momentum-compensated tear aiming (predictive shooting while walking).
  - Melee kiting maintaining optimal damage sweet-spots.
- **🚪 Room & Dungeon Navigation**:
  - Automatic detection of room clear events.
  - Door identification (Normal, Item Room, Shop, Boss Room).
  - Item pedestal and pickup collection.
- **🎮 Standard Gymnasium Interface**:
  - `gymnasium.Env` compliance for training PPO, SAC, DreamerV3, and DQN algorithms.
- **🖥️ Non-intrusive Native Desktop Integration**:
  - Direct Win32 / DirectInput key emulation.
  - Zero binary modification required.

---

## 🏗️ Architecture

```
                                 [ Isaac Window (60 FPS) ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Screen Capture ]                            [ Input Simulator ]
             (mss / Win32 GDI)                             (SendInput Scancodes)
                       │                                           ▲
                       ▼                                           │
             [ Vision Detector ]                                   │
      (Isaac / Enemies / Tears / HUD)                              │
                       │                                           │
                       ├────────────────────────┐                  │
                       ▼                        ▼                  │
            [ Potential-Field Combat ]   [ Macro Dungeon ]         │
             (Micro Dodging & Aiming)     (Room Navigation)        │
                       │                        │                  │
                       └────────────┬───────────┘                  │
                                    ▼                              │
                       [ Gymnasium Environment ] ──────────────────┘
                                    │
                       [ Deep RL Policy (PPO/SAC) ]
```

---

## 🚀 Quick Start

### 1. Requirements & Setup

```bash
git clone https://github.com/DeepArcadia/IsaacRL.git
cd IsaacRL
pip install -r requirements.txt
```

### 2. Launch Isaac
Start *The Binding of Isaac: Repentance+* through Steam in windowed or borderless mode.

### 3. Run the Autonomous Agent

```bash
# Run the reactive combat autopilot with live visualizer
python scripts/run_agent.py --mode autopilot --debug
```

### 4. Train with Reinforcement Learning

```bash
# Train PPO agent on top of the Gymnasium environment
python scripts/train_rl.py --algorithm ppo --timesteps 100000
```

---

## 📜 License
MIT License. Developed with ❤️ by [DeepArcadia](https://github.com/DeepArcadia).
