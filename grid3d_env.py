import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Grid3DEnv(gym.Env):
    def __init__(self, M=10, obstacle_ratio=0.5, max_steps=500):
        super(Grid3DEnv, self).__init__()
        self.M = M
        self.max_steps = max_steps
        self.obstacle_ratio = obstacle_ratio

        # Define action space: 6 directions (x,y,z +/-1)
        self.action_space = spaces.Discrete(6)

        # Observation space: flattened view grid + agent and goal position (one-hot or int tuple)
        self.view_shape = (M, M, M)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.view_shape, dtype=np.uint8)

        self._generate_map()
        self.reset()

    def _generate_map(self):
        self.obstacle_map = np.random.choice([0, 1], size=(self.M, self.M, self.M), p=[1 - self.obstacle_ratio, self.obstacle_ratio])

    def _random_empty_pos(self):
        empties = np.argwhere(self.obstacle_map == 0)
        idx = np.random.choice(len(empties))
        return tuple(empties[idx])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Optional: for Gym compatibility
        self.current_step = 0
        self.agent_pos = self._random_empty_pos()
        self.goal_pos = self._random_empty_pos()
        while self.goal_pos == self.agent_pos:
            self.goal_pos = self._random_empty_pos()
        return self._get_obs(), {}



    def _get_obs(self):
        # Simple observable map: 1 for visible 0s, -1 for hidden, 0 for blocked
        visible = np.full_like(self.obstacle_map, fill_value=-1)

        ax, ay, az = self.agent_pos
        for x in range(self.M):
            for y in range(self.M):
                for z in range(self.M):
                    if self._has_line_of_sight(self.agent_pos, (x, y, z)):
                        visible[x, y, z] = self.obstacle_map[x, y, z]

        return visible.astype(np.uint8)

    def step(self, action):
        self.current_step += 1
        moves = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        dx, dy, dz = moves[int(action)]  # Convert to int if it's numpy type

        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy, self.agent_pos[2] + dz)
        if self._valid_pos(new_pos):
            self.agent_pos = new_pos

        done = self.agent_pos == self.goal_pos or self.current_step >= self.max_steps

        if self.agent_pos == self.goal_pos:
            reward = 1000 - self.current_step * 5  # Higher reward for faster completion
        else:
            reward = -1  # Small penalty per step

        return self._get_obs(), reward, done, False, {}

    def _valid_pos(self, pos):
        x, y, z = pos
        return (0 <= x < self.M) and (0 <= y < self.M) and (0 <= z < self.M) and (self.obstacle_map[x, y, z] == 0)

    def _has_line_of_sight(self, start, end):
        # Bresenham's algorithm in 3D (simplified)
        x1, y1, z1 = start
        x2, y2, z2 = end
        x1, y1, z1 = int(x1), int(y1), int(z1)
        x2, y2, z2 = int(x2), int(y2), int(z2)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                if self.obstacle_map[x1, y1, z1] == 1:
                    return False
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                if self.obstacle_map[x1, y1, z1] == 1:
                    return False
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                if self.obstacle_map[x1, y1, z1] == 1:
                    return False
        return True
