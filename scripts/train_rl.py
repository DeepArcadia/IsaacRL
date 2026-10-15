"""PPO Reinforcement Learning Training Script for IsaacRL."""

import argparse
import os
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from isaacrl.env.isaac_env import IsaacEnv


class ActorCritic(nn.Module):
    """Convolutional Actor-Critic Network for visual Isaac gameplay."""

    def __init__(self, obs_shape=(3, 84, 84)):
        super().__init__()
        # Vision backbone (CNN)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # MultiDiscrete Actor Heads: Move X (3), Move Y (3), Shoot (5)
        self.actor_move_x = nn.Linear(512, 3)
        self.actor_move_y = nn.Linear(512, 3)
        self.actor_shoot = nn.Linear(512, 5)

        # Critic Head (Value function)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        feat = self.conv(x)
        return feat

    def get_action_and_value(self, x, action=None):
        feat = self.forward(x)
        logits_mx = self.actor_move_x(feat)
        logits_my = self.actor_move_y(feat)
        logits_sh = self.actor_shoot(feat)

        dist_mx = Categorical(logits=logits_mx)
        dist_my = Categorical(logits=logits_my)
        dist_sh = Categorical(logits=logits_sh)

        if action is None:
            a_mx = dist_mx.sample()
            a_my = dist_my.sample()
            a_sh = dist_sh.sample()
            action = torch.stack([a_mx, a_my, a_sh], dim=-1)

        log_prob = (
            dist_mx.log_prob(action[..., 0])
            + dist_my.log_prob(action[..., 1])
            + dist_sh.log_prob(action[..., 2])
        )
        entropy = dist_mx.entropy() + dist_my.entropy() + dist_sh.entropy()
        value = self.critic(feat)
        return action, log_prob, entropy, value


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent on The Binding of Isaac")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save weights")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing PPO Training on Device: {device}")

    env = IsaacEnv(obs_size=(84, 84))
    model = ActorCritic(obs_shape=(3, 84, 84)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    obs, _ = env.reset(options={"restart": False})
    print("✅ Environment ready. Beginning interaction loop...")

    try:
        for step in range(args.timesteps):
            # Format observation: (H, W, C) -> (1, C, H, W) normalized to [0, 1]
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                action, log_prob, _, val = model.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            if step % 20 == 0:
                print(
                    f"Step: {step:5d} | Reward: {reward:+.3f} | "
                    f"Enemies: {info.get('enemies', 0)} | FPS: {info.get('fps', 0):.1f}"
                )

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset(options={"restart": True})

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        ckpt_path = os.path.join(args.checkpoint_dir, "isaac_ppo_final.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved model checkpoint to: {ckpt_path}")
        env.close()


if __name__ == "__main__":
    main()
