from envs.nav_env import NavEnv
import numpy as np

env = NavEnv()
obs, _ = env.reset()

for i in range(30):
    action = np.array([0.5, 0.2])
    obs, r, done, trunc, info = env.step(action)
    print(i, "min ray:", obs[:12].min(), "max ray:", obs[:12].max(), "collision:", info.get("collision", False))
    if done:
        obs, _ = env.reset()

