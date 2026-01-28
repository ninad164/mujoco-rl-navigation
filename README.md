🚗 Reinforcement Learning Navigation in MuJoCo

This project demonstrates end-to-end autonomous navigation using Reinforcement Learning (PPO) in a simulated 2D environment built with MuJoCo.

A mobile robot must navigate to a goal position while avoiding randomized obstacles, using only LiDAR-style ray sensor inputs. Performance is compared against a hand-crafted reactive baseline controller.

🎯 Problem Setup

Each episode randomizes:

Robot start position

Goal location

6 static obstacles placed in the arena

The robot receives:

36 LiDAR-style distance rays

Distance to goal

Angle to goal

It outputs:

Linear velocity

Angular velocity

This creates a partial observability + obstacle avoidance + goal-reaching task.

🧠 Methods
Baseline Controller (Rule-Based)

A reactive controller that:

Turns toward the goal

Slows or steers away when LiDAR detects close obstacles

Has no memory or planning

Reinforcement Learning (PPO)

A Proximal Policy Optimization (PPO) agent trained using:

Stable-Baselines3

8 parallel simulation environments

800,000 timesteps of training

Reward shaping based on:

Progress toward goal

Collision penalty

Goal completion bonus

The learned policy discovers navigation strategies that outperform the reactive baseline.

📊 Results (100 Evaluation Episodes)
Metric	Baseline	PPO	Improvement
Success Rate	0.28	0.71	+153.6%
Median Time to Goal	8.74 s	3.30 s	62.2% faster
Collision Rate	0.20	0.21	Slightly higher (+1%)
Key Takeaway

The RL policy learns efficient obstacle negotiation and path planning, dramatically improving goal-reaching performance while maintaining similar collision rates.

🎥 Visualizations

The project includes rendered rollouts showing:

Baseline navigation behavior

PPO-learned navigation behavior

These highlight how the RL agent:

Takes smoother trajectories

Avoids dead-ends more effectively

Reaches the goal significantly faster

🏗️ Project Structure
mujoco_rl_nav/
│
├── envs/
│   └── nav_env.py          # Custom MuJoCo navigation environment
│
├── controllers/
│   └── baseline.py         # Rule-based controller
│
├── scripts/
│   ├── train_ppo.py        # Train PPO agent
│   ├── eval_baseline.py    # Evaluate baseline controller
│   ├── eval_ppo.py         # Evaluate PPO agent
│   ├── compare_results.py  # Metric comparison
│   └── render_rollout.py   # Generate animation videos
│
└── arena.xml               # MuJoCo environment definition

⚙️ Installation
conda create -n mujoco_rl python=3.10
conda activate mujoco_rl

pip install mujoco gymnasium stable-baselines3[extra] numpy pandas imageio


MuJoCo 2.3+ is required.

🚀 How to Run
Train PPO Agent
python -m scripts.train_ppo

Evaluate Baseline
python -m scripts.eval_baseline --seed 0

Evaluate PPO
python -m scripts.eval_ppo --seed 0

Compare Results
python -m scripts.compare_results

Render Example Episodes
python -m scripts.render_rollout

🧩 Skills Demonstrated

Reinforcement Learning (PPO)

Continuous control policy learning

MuJoCo simulation and rendering

Sensor simulation via ray casting

Custom Gymnasium environment design

Performance benchmarking vs rule-based systems

Parallelized RL training

💡 Future Improvements

Add moving (dynamic) obstacles

Incorporate memory (LSTM policy)

Add curriculum learning for progressively harder maps

Penalize near-collisions for safer policies

📌 Summary

This project shows how learning-based control outperforms hand-coded rules in complex navigation tasks, especially when environments are randomized and unpredictable — a key requirement in real-world robotics and autonomous systems.