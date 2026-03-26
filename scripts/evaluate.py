import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from custom_env import CustomStocksEnv

def evaluate_agent():
    print("Loading datasets...")
    df = pd.read_csv("data/AAPL_with_indicators.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # We trained on the first ~80%. We evaluate on the last ~20%.
    train_size = int(len(df) * 0.8)
    # The start index for testing must have a buffer for the window size
    window_size = 20
    test_frame = (train_size, len(df))
    print(f"Testing on data from index: {test_frame}")

    def make_env():
        return CustomStocksEnv(df=df, window_size=window_size, frame_bound=test_frame)
    
    env = DummyVecEnv([make_env])
    
    # Load previously saved normalization statistics (DO NOT update them during testing)
    env = VecNormalize.load("models/vec_normalize.pkl", env)
    env.training = False 
    env.norm_reward = False

    print("Loading RecurrentPPO Model...")
    model = RecurrentPPO.load("models/aapl_recurrent_ppo", env=env)
    
    observation = env.reset()
    # LSTM states initialization
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    print("Running evaluation...")
    while True:
        action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_start)
        observation, rewards, done, info = env.step(action)
        episode_start = done
        # Usually DummyVecEnv auto-resets on done
        if done[0]:
            print(f"Final Info: {info[0]}")
            break

    # Re-instantiating a single environment for plotting because DummyVecEnv abstracts some rendering
    raw_env = make_env()
    # We need to run through the raw_env normally to plot the single actions
    raw_env.reset()
    observation = env.reset() # use normalized obs for prediction
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    
    while True:
        action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_start)
        # Advance raw_env to plot correctly
        raw_env.step(action[0])
        # Advance normalized env for model
        observation, rewards, done, info = env.step(action)
        episode_start = done
        if done[0]:
            break
            
    plt.figure(figsize=(15,6))
    raw_env.render_all()
    plt.title("LSTM RL Agent Evaluation - Test Set (AAPL)")
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/lstm_evaluation.png")
    print("Evaluation Complete. Plot saved to reports/figures/lstm_evaluation.png")


if __name__ == "__main__":
    evaluate_agent()
