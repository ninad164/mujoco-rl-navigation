import os
from matplotlib.pyplot import step
import numpy as np
import imageio.v2 as imageio

import mujoco
from stable_baselines3 import PPO

from envs.nav_env import NavEnv
from controllers.baseline import BaselineController


def rollout_frames(env, policy="baseline", max_steps=1500, width=640, height=640):
    frames = []

    if policy == "baseline":
        ctrl = BaselineController(n_rays=env.n_rays)
        model = None
    else:
        ctrl = None
        model = PPO.load("ppo_nav_rand6.zip", device="cpu")

    renderer = mujoco.Renderer(env.model, width=width, height=height)

    obs, _ = env.reset(seed=0)

    reached = False
    collided = False
    steps = 0

    for _ in range(max_steps):
        # Choose action
        if ctrl is not None:
            action = ctrl.act(obs)
        else:
            action, _ = model.predict(obs, deterministic=True)

        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # Render frame
        renderer.update_scene(env.data, camera="topdown")
        frame = renderer.render()
        frames.append(frame)

        # Stop conditions for video
        if info.get("goal_reached", False):
            reached = True
            break

        if info.get("collision", False):
            collided = True
            break

        # Important: ignore truncated (timeout) so baseline can keep trying.
        # We do NOT break on (terminated or truncated) anymore.

    print(f"[{policy.upper()}] Rollout ended: steps={steps}, reached={reached}, collided={collided}")

    renderer.close()
    return frames

def save_mp4(frames, out_path, fps=30):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved: {out_path}  ({len(frames)} frames)")

def side_by_side(frames_left, frames_right):
    n = min(len(frames_left), len(frames_right))
    combo = []
    for i in range(n):
        combo.append(np.concatenate([frames_left[i], frames_right[i]], axis=1))
    return combo

if __name__ == "__main__":
    env = NavEnv()

    print("Rendering baseline episode...")
    frames_b = rollout_frames(env, policy="baseline")
    imageio.mimsave("baseline.mp4", frames_b, fps=30)

    print("Rendering PPO episode...")
    frames_p = rollout_frames(env, policy="ppo")
    imageio.mimsave("ppo.mp4", frames_p, fps=30)
    
    combo = side_by_side(frames_b, frames_p)
    imageio.mimsave("baseline_vs_ppo.mp4", combo, fps=20)
    print("Saved baseline_vs_ppo.mp4")

    #print("Saved baseline.mp4 and ppo.mp4")

