import time
import numpy as np
import pandas as pd

from envs.nav_env import NavEnv
from controllers.baseline import BaselineController


def run_eval(n_episodes=100, seed=0):
    np.random.seed(seed)

    env = NavEnv()
    ctrl = BaselineController(n_rays=env.n_rays)

    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        t0 = time.time()

        collided = False
        reached = False
        steps = 0

        while True:
            action = ctrl.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if info.get("collision", False):
                collided = True
            if info.get("goal_reached", False):
                reached = True

            if terminated or truncated:
                break

        dt = time.time() - t0
        rows.append({
            "episode": ep,
            "steps": steps,
            "time_sec": dt,
            "collision": int(collided),
            "success": int(reached),
        })

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame):
    n = len(df)
    collision_rate = df["collision"].mean()
    success_rate = df["success"].mean()

    # time-to-goal only for successful episodes
    success_df = df[df["success"] == 1]
    if len(success_df) > 0:
        median_time = float(success_df["time_sec"].median())
        median_steps = float(success_df["steps"].median())
    else:
        median_time = None
        median_steps = None

    print("\n=== BASELINE RESULTS ===")
    print(f"Episodes: {n}")
    print(f"Collision rate: {collision_rate:.3f}")
    print(f"Success rate:   {success_rate:.3f}")
    if median_time is not None:
        print(f"Median time-to-goal (success only): {median_time:.2f} s")
        print(f"Median steps-to-goal (success only): {median_steps:.1f}")
    else:
        print("No successful episodes; cannot compute time-to-goal.")

    return {
        "episodes": n,
        "collision_rate": collision_rate,
        "success_rate": success_rate,
        "median_time_sec_success": median_time,
        "median_steps_success": median_steps
    }


if __name__ == "__main__":
    df = run_eval(n_episodes=100, seed=0)
    summarize(df)

    # Save CSV
    df.to_csv("baseline_metrics.csv", index=False)
    print("\nSaved baseline_metrics.csv")
