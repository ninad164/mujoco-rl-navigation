import numpy as np

class BaselineController:
    """
    Simple reactive controller:
      - Turn toward goal.
      - If front rays are close, steer away from obstacle.
    """

    def __init__(self, n_rays=36):
        self.n_rays = n_rays

    def act(self, obs: np.ndarray) -> np.ndarray:
        rays = obs[:self.n_rays]
        goal_dist = float(obs[self.n_rays])          # 0..1
        goal_bearing_n = float(obs[self.n_rays + 1]) # -1..1

        # bearing in radians
        bearing = goal_bearing_n * np.pi

        # FRONT obstacle metric
        front = int(self.n_rays / 2)
        win = 4
        front_slice = rays[front - win: front + win + 1]
        min_front = float(np.min(front_slice))

        # Potential-field style:
        # - attractive: turn toward goal
        # - repulsive: turn away if obstacle close in front
        w_goal = 2.5 * np.clip(bearing, -1.0, 1.0)

        w_avoid = 0.0
        if min_front < 0.5:
            left = float(np.mean(rays[front+1: front+1+8]))
            right = float(np.mean(rays[front-8: front]))
            # steer toward clearer side
            w_avoid = 2.0 if left > right else -2.0
            # stronger avoid when closer
            w_avoid *= (0.5 - min_front) / 0.5

        w = w_goal + w_avoid
        w = float(np.clip(w, -2.0, 2.0))

        # speed: slow down if turning hard or obstacle close
        turn_penalty = min(abs(w) / 2.0, 1.0)
        obs_penalty = max(0.0, (0.5 - min_front) / 0.5) if min_front < 0.5 else 0.0

        v = 1.0 - 0.6 * turn_penalty - 0.8 * obs_penalty
        v = float(np.clip(v, 0.05, 1.0))

        # slow down near goal
        if goal_dist < 0.15:
            v = min(v, 0.2)

        return np.array([v, w], dtype=np.float32)
