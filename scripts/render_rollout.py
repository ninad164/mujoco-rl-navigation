import os
import numpy as np
import imageio.v2 as imageio

import mujoco
from stable_baselines3 import PPO

from envs.nav_env import NavEnv
from controllers.baseline import BaselineController


def rollout_frames(env: NavEnv, policy="baseline", max_steps=500, width=640, height=480,
                   force_start=True, require_success=False, max_tries=50):
    """
    policy: "baseline" or path to PPO zip (e.g. "ppo_nav.zip")
    force_start: if True, start from a fixed pose far from the goal
    require_success: if True, keep sampling episodes until goal_reached
    """
    # Prepare controller / model
    if policy == "baseline":
        ctrl = BaselineController(n_rays=env.n_rays)
        def act(o):
            return ctrl.act(o)
    else:
        model = PPO.load(policy, device="cpu")
        def act(o):
            a, _ = model.predict(o, deterministic=True)
            return a

    renderer = mujoco.Renderer(env.model, height=height, width=width)

    for attempt in range(max_tries):
        obs, _ = env.reset()

        # Force a consistent, far start so videos are meaningful
        if force_start:
            x0, y0 = -1.8, -1.8
            theta0 = np.arctan2(env.goal[1] - y0, env.goal[0] - x0)
            env.data.qpos[0] = x0
            env.data.qpos[1] = y0
            env.data.qpos[2] = theta0
            mujoco.mj_forward(env.model, env.data)
            env.prev_dist = env._goal_distance()
            obs = env._get_obs()

        frames = []
        reached = False

        for _ in range(max_steps):
            action = act(obs)
            obs, r, terminated, truncated, info = env.step(action)

            renderer.update_scene(env.data, camera=-1)  # <-- important
            frames.append(renderer.render())

            if info.get("goal_reached", False):
                reached = True

            if terminated or truncated:
                break

        # If we require success (baseline), retry until we get a successful episode
        if not require_success or reached:
            return frames

    # If we couldn't find a success episode, return the last attempt
    return frames



def save_mp4(frames, out_path, fps=30):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    env = NavEnv()

    # Baseline: require a successful episode so it actually reaches the goal
    frames_b = rollout_frames(env, policy="baseline", max_steps=600,
                              force_start=True, require_success=True)
    save_mp4(frames_b, "results/baseline.mp4", fps=30)

    # PPO: fixed far start so it doesn't end instantly
    frames_p = rollout_frames(env, policy="ppo_nav.zip", max_steps=600,
                              force_start=True, require_success=False)
    save_mp4(frames_p, "results/ppo.mp4", fps=30)
