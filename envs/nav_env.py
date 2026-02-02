import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class NavEnv(gym.Env):
    """
    Differential-drive style navigation (kinematic) in a MuJoCo arena.

    Observation (float32, shape=(38,)):
      - 36 lidar rays normalized to [0,1] (0=very close, 1=max_range)
      - goal distance normalized (dist/10)
      - goal bearing normalized (angle/pi)

    Action (float32, shape=(2,)):
      - v in [0, 1]  (m/s)
      - w in [-1, 1] (rad/s)

    Termination:
      - goal reached (dist < 0.3)
      - collision (robot_body contacts something)
      - timeout (max_steps)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("envs/assets/arena.xml")
        self.data = mujoco.MjData(self.model)

        self.obs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs3"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs4"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs5"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obs6")       
        ]
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")

        # IDs for fast lookup
        self.robot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_body")
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")

        # Episode settings
        self.max_steps = 500
        self.step_count = 0

        # Goal
        self.goal = np.array([2.0, 2.0], dtype=np.float64)

        # For progress reward
        self.prev_dist = None

        # Lidar config
        self.n_rays = 36            # num of rays
        self.max_range = 6.0

        # Spaces
        obs_dim = self.n_rays + 2
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Sample a start
        start = np.array([-2.2, -2.2], dtype=np.float64)

        # Sample a goal
        while True:
            goal = np.array([2.2, 2.2], dtype=np.float64)
            if np.linalg.norm(goal - start) > 2.5:                         # ensures distance
                break
        self.goal = goal

        # print("start:", start, "goal:", self.goal)

        # set robot pose (face goal)
        self.data.qpos[0] = float(start[0])
        self.data.qpos[1] = float(start[1])
        theta_to_goal = np.arctan2(self.goal[1] - start[1], self.goal[0] - start[0])
        self.data.qpos[2] = float(theta_to_goal)

        # clear velocities
        if self.data.qvel is not None:
            self.data.qvel[:] = 0

        # Move goal marker (green dot) to the sampled goal
        self.model.body_pos[self.goal_body_id, 0] = float(self.goal[0])
        self.model.body_pos[self.goal_body_id, 1] = float(self.goal[1])

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = self._goal_distance()

        obs = self._get_obs()
        return obs, {}

    def _goal_distance(self) -> float:
        pos = np.array(self.data.qpos[0:2], dtype=np.float64)
        return float(np.linalg.norm(self.goal - pos))

    def _goal_bearing(self) -> float:
        """Bearing to goal in robot frame, wrapped to [-pi, pi]."""
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        theta = float(self.data.qpos[2])

        global_angle = np.arctan2(self.goal[1] - y, self.goal[0] - x)
        bearing = global_angle - theta
        return float(np.arctan2(np.sin(bearing), np.cos(bearing)))

    def _lidar(self) -> np.ndarray:
        """
        Raycast lidar using mj_ray.
        Returns normalized distances in [0,1], shape (n_rays,).
        """
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        theta = float(self.data.qpos[2])

        origin = np.array([x, y, 0.08], dtype=np.float64)
        rays = np.zeros(self.n_rays, dtype=np.float32)

        angles = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)

        # Output buffer for hit geom id (required by this MuJoCo binding)
        geomid_out = np.zeros(1, dtype=np.int32)

        for i, a in enumerate(angles):
            ang = theta + a
            direction = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)

            # Returns distance only; writes hit geom id to geomid_out[0]
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                None,               # geomgroup mask
                1,                  # include static geoms
                self.robot_body_id, # exclude robot body
                geomid_out
            )

            # If no hit, MuJoCo returns negative distance
            if dist < 0:
                dist = self.max_range

            dist = min(dist, self.max_range)
            rays[i] = dist / self.max_range         # scale distance to [0,1]

        return rays

    def _check_collision(self) -> bool:
        """True only if robot_body touches something other than the floor."""
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.robot_geom_id or c.geom2 == self.robot_geom_id:
                other = c.geom2 if c.geom1 == self.robot_geom_id else c.geom1
                if other != floor_geom_id:
                    return True
        return False

    def _get_obs(self) -> np.ndarray:
        rays = self._lidar()

        dist = self._goal_distance()
        bearing = self._goal_bearing()

        # Normalize dist and bearing
        dist_n = np.clip(dist / 10.0, 0.0, 1.0)
        bearing_n = np.clip(bearing / np.pi, -1.0, 1.0)

        obs = np.concatenate([rays, [dist_n, bearing_n]]).astype(np.float32)
        return obs

    def step(self, action):
        # Ensure correct dtype and clip to action space
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        w = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))

        # Explicit kinematic integration (more reliable for this simple env)
        dt = float(self.model.opt.timestep)

        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        theta = float(self.data.qpos[2])

        # differential model
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt

        # wrap theta
        theta = float(np.arctan2(np.sin(theta), np.cos(theta)))

        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = theta

        mujoco.mj_forward(self.model, self.data)  # update contacts, etc.

        self.step_count += 1

        # Reward: progress toward goal
        dist = self._goal_distance()
        reward = float(self.prev_dist - dist)
        self.prev_dist = dist

        # Termination checks
        collision = self._check_collision()
        reached_goal = dist < 0.5                   # robot doesn't actually reach the goal body
        timeout = self.step_count >= self.max_steps # happens in baseline case

        terminated = False
        info = {}

        if collision:
            reward -= 1.0
            terminated = True
            info["collision"] = True

        if reached_goal:
            reward += 10.0
            terminated = True
            info["goal_reached"] = True

        truncated = bool(timeout and not terminated)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
