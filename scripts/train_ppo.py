from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.nav_env import NavEnv

if __name__ == "__main__":
    # 8 parallel envs = more diverse layouts + faster data collection
    env = make_vec_env(NavEnv, n_envs=8, seed=0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=1024,      # per-env rollout; 1024 * 8 = 8192 steps per update
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=800_000)
    model.save("ppo_nav_rand6")
    print("Saved model: ppo_nav_rand6.zip")
