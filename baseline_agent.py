import gymnasium as gym
import gym_anytrading
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

# 1. Create Environment
# 'stocks-v0' is a default environment provided by gym-anytrading
# We'll use a small window for the baseline
env = gym.make('stocks-v0', frame_bound=(50, 200), window_size=10)

print("Environment created.")
print("> Action Space:", env.action_space)
print("> Observation Space:", env.observation_space)

# 2. Setup Model
# A2C is an Actor-Critic algorithm, 'MlpPolicy' is a Multi-Layer Perceptron
model = A2C('MlpPolicy', env, verbose=1)

# 3. Train the agent
print("Starting training...")
model.learn(total_timesteps=2000)
print("Training finished.")

# 4. Save the model
model.save("models/a2c_stocks_baseline")

# 5. Evaluate
print("Evaluating agent...")
observation, info = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print("Final Info:", info)
        break

# 6. Plot results
plt.figure(figsize=(15, 6))
plt.cla()
env.unwrapped.render_all()
plt.title("Baseline A2C Trading Agent - Stocks")
plt.savefig("reports/figures/baseline_performance.png")
print("Performance plot saved to reports/figures/baseline_performance.png")
