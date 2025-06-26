import os
from stable_baselines3 import DQN
from grid3d_env import Grid3DEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

M = 10
MODEL_PATH = "dqn_grid3d_10x10x10"

def make_env():
    def _init():
        env = Grid3DEnv(M=M)
        env = Monitor(env)
        return env
    return _init

env = DummyVecEnv([make_env()])

if os.path.exists(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = DQN.load(MODEL_PATH, env=env)
else:
    print("Creating new model...")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4, device="cuda")  # or device="cpu"

print("Using device:", model.device)

print("Starting training...")
model.learn(total_timesteps=50000, reset_num_timesteps=False)
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
