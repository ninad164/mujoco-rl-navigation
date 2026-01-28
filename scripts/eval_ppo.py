import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from envs.nav_env import NavEnv


def run_eval(model_path="ppo_nav_rand6.zip", n_episodes=100, seed=0):
    # np.random.seed(seed)

    env = NavEnv()
    model = PPO.load(model_path, device="cpu")

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        collided = False
        reached = False
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if info.get("collision", False):
                collided = True
            if info.get("goal_reached", False):
                reached = True

            if terminated or truncated:
                break

        sim_time = steps * float(env.model.opt.timestep)

        rows.append({
            "episode": ep,
            "steps": steps,
            "time_sec": sim_time,
            "collision": int(collided),
            "success": int(reached),
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    n = len(df)
    collision_rate = df["collision"].mean()
    success_rate = df["success"].mean()

    success_df = df[df["success"] == 1]
    if len(success_df) > 0:
        median_time = float(success_df["time_sec"].median())
        median_steps = float(success_df["steps"].median())
    else:
        median_time = None
        median_steps = None

    print("\n=== PPO RESULTS ===")
    print(f"Episodes: {n}")
    print(f"Collision rate: {collision_rate:.3f}")
    print(f"Success rate:   {success_rate:.3f}")
    if median_time is not None:
        print(f"Median time-to-goal (success only): {median_time:.2f} s")
        print(f"Median steps-to-goal (success only): {median_steps:.1f}")
    else:
        print("No successful episodes; cannot compute time-to-goal.")


if __name__ == "__main__":
    df = run_eval(model_path="ppo_nav.zip", n_episodes=100, seed=0)
    summarize(df)
    df.to_csv("ppo_metrics.csv", index=False)
    print("\nSaved ppo_metrics.csv")
