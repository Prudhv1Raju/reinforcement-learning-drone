import numpy as np
import torch

# ✅ Import from your other notebooks/files
from envi_wrap import DroneEnv
from sac import SACAgent
from replay import ReplayBuffer

# ------------------------
# Hyperparameters
# ------------------------
NUM_EPISODES = 500
MAX_STEPS = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 3e-4

# ------------------------
# Initialize environment
# ------------------------
env = DroneEnv()

# ------------------------
# Initialize agent and buffer
# ------------------------
agent = SACAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=LR)
buffer = ReplayBuffer(max_size=100000)

# ------------------------
# Training loop
# ------------------------
for episode in range(NUM_EPISODES):
    state = env.reset()
    
    episode_reward = 0
    done = False
    step = 0
    
    while not done and step < MAX_STEPS:
        # Select action from policy (with noise for exploration)
        action = agent.select_action(state)
        
        # Apply action
        next_state, reward, done, info = env.step(action)

        # Store transition
        buffer.add(state, action, reward, next_state, done)
        episode_reward += reward

        state = next_state
        step += 1

        # Update policy if enough samples
        if len(buffer) > BATCH_SIZE:
            for _ in range(1):  # Tune how many updates per step if needed
                agent.update(buffer, batch_size=BATCH_SIZE, gamma=GAMMA)

    print(f"Episode {episode+1} Reward: {episode_reward}")

print("✅ Training finished!")