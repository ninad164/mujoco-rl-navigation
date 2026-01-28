import pandas as pd


def pct_improve(base, new, higher_is_better=True):
    if base == 0:
        return None
    if higher_is_better:
        return 100.0 * (new - base) / base
    return 100.0 * (base - new) / base


def main():
    b = pd.read_csv("baseline_metrics.csv")
    p = pd.read_csv("ppo_metrics.csv")

    base_collision = float(b["collision"].mean())
    ppo_collision = float(p["collision"].mean())

    base_success = float(b["success"].mean())
    ppo_success = float(p["success"].mean())

    b_succ = b[b["success"] == 1]
    p_succ = p[p["success"] == 1]

    base_med_time = float(b_succ["time_sec"].median()) if len(b_succ) else None
    ppo_med_time = float(p_succ["time_sec"].median()) if len(p_succ) else None

    print("\n=== BASELINE vs PPO ===")
    print(f"Collision rate: {base_collision:.3f} -> {ppo_collision:.3f} | "
          f"improvement: {pct_improve(base_collision, ppo_collision, higher_is_better=False):.1f}%")
    print(f"Success rate:   {base_success:.3f} -> {ppo_success:.3f} | "
          f"improvement: {pct_improve(base_success, ppo_success, higher_is_better=True):.1f}%")

    if base_med_time is not None and ppo_med_time is not None:
        print(f"Median time-to-goal (success): {base_med_time:.2f}s -> {ppo_med_time:.2f}s | "
              f"improvement: {pct_improve(base_med_time, ppo_med_time, higher_is_better=False):.1f}%")
    else:
        print("Median time-to-goal: not available for one of the methods (no successes).")


if __name__ == "__main__":
    main()
