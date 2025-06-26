import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grid3d_env import Grid3DEnv
from stable_baselines3 import DQN

# Load the trained model
model = DQN.load("dqn_grid3d_10x10x10")

# Create the environment
env = Grid3DEnv(M=10)

obs = env.reset()
done = False
positions = [env.agent_pos]

while not done:
    action, _states = model.predict(obs)
    action = int(action)  # Convert numpy array action to int
    obs, reward, done, info = env.step(action)
    positions.append(env.agent_pos)

positions = np.array(positions)

# Plotting the agent path in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot obstacles as red cubes
obstacle_positions = np.argwhere(env.grid == 1)
ax.scatter(obstacle_positions[:,0], obstacle_positions[:,1], obstacle_positions[:,2], c='red', marker='s', s=100, label='Obstacles')

# Plot agent path as blue dots
ax.plot(positions[:,0], positions[:,1], positions[:,2], c='blue', marker='o', label='Agent path')

# Start and goal points
ax.scatter(env.start_pos[0], env.start_pos[1], env.start_pos[2], c='green', s=150, label='Start')
ax.scatter(env.goal_pos[0], env.goal_pos[1], env.goal_pos[2], c='purple', s=150, label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Agent path in 3D grid')
ax.legend()
plt.show()
