import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from custom_env import CustomStocksEnv
import time

def run_experiment(learning_rate, window_size, df, test_frame):
    print(f"--- Running experiment: LR={learning_rate}, Window={window_size} ---")
    
    train_size = int(len(df) * 0.8)
    train_frame = (window_size, train_size)
    
    def make_train_env():
        return CustomStocksEnv(df=df, window_size=window_size, frame_bound=train_frame)
    
    env = DummyVecEnv([make_train_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=0,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64
    )
    
    start_time = time.time()
    # Shorter timestep just for experiment testing/demonstration purposes
    model.learn(total_timesteps=15000)
    training_time = time.time() - start_time
    
    # Evaluation
    def make_test_env():
        return CustomStocksEnv(df=df, window_size=window_size, frame_bound=test_frame)
    
    test_env = DummyVecEnv([make_test_env])
    # Sync normalization stats
    test_env.obs_rms = env.obs_rms
    test_env.training = False
    test_env.norm_reward = False
    
    observation = test_env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    total_rewards = 0
    total_profit = 0.0
    
    while True:
        action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_start)
        observation, rewards, done, info = test_env.step(action)
        total_rewards += rewards[0]
        episode_start = done
        if done[0]:
            total_profit = info[0]['total_profit']
            break
            
    # Save a plot for this specific run
    plt.figure(figsize=(10,4))
    raw_env = make_test_env()
    raw_env.reset()
    observation = test_env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    while True:
        action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_start)
        raw_env.step(action[0])
        observation, rewards, done, info = test_env.step(action)
        episode_start = done
        if done[0]:
            break
            
    raw_env.render_all()
    plt.title(f"LR: {learning_rate} | Window: {window_size} | Profit: {total_profit:.2f}")
    plt.tight_layout()
    plot_path = f"reports/figures/exp_lr_{learning_rate}_win_{window_size}.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "Learning Rate": learning_rate,
        "Window Size": window_size,
        "Total Profit": total_profit,
        "Total Reward": total_rewards,
        "Training Time (s)": round(training_time, 2)
    }

def main():
    os.makedirs("reports/figures", exist_ok=True)
    df = pd.read_csv("data/AAPL_with_indicators.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    train_size = int(len(df) * 0.8)
    # Give it a safe buffer for test frame
    test_frame = (train_size, len(df))
    
    learning_rates = [0.001, 0.0003]
    window_sizes = [10, 20]
    
    results = []
    
    for lr in learning_rates:
        for ws in window_sizes:
            res = run_experiment(lr, ws, df, test_frame)
            results.append(res)
            
    results_df = pd.DataFrame(results)
    print("\n--- Experiment Results ---")
    print(results_df.to_string(index=False))
    
    results_df.to_csv("reports/experiment_results.csv", index=False)
    print("\nSaved experiment results to reports/experiment_results.csv")

if __name__ == "__main__":
    main()
