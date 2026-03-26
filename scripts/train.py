import os
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from custom_env import CustomStocksEnv
import warnings
warnings.filterwarnings('ignore')

def train_agent():
    print("Loading dataset...")
    df = pd.read_csv("data/AAPL_with_indicators.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Let's split train and test data. 
    # E.g., Train on first 80%, Test on last 20%
    train_size = int(len(df) * 0.8)
    
    # gym-anytrading uses 'window_size' and 'frame_bound'
    window_size = 20
    train_frame = (window_size, train_size)
    
    print(f"Dataset size: {len(df)}. Training on: {train_frame}")

    # 1. Create Train Environment
    def make_env():
        return CustomStocksEnv(df=df, window_size=window_size, frame_bound=train_frame)
    
    env = DummyVecEnv([make_env])
    
    # 2. Normalize observation features (essential for neural networks)
    # the RL agent performs much better if features like MACD/prices are normalized.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 3. Define the Recurrent PPO model (LSTM)
    print("Initializing Recurrent PPO (LSTM) Agent...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=dict(
            # Define architecture: shared layers, then LSTM, then actor/critic heads.
            # net_arch=[dict(pi=[64, 64], vf=[64, 64])] for sb3. In sb3-contrib RecurrentPPO,
            # Lstm parameters are handled natively. We can specify layer sizes before/after LSTM.
            # A simple default is good enough.
        )
    )
    
    # 4. Train
    print("Starting Training...")
    # 50,000 timesteps is a quick baseline for a university assignment.
    # In real scenarios this would be millions.
    model.learn(total_timesteps=50000)
    
    # 5. Save model and normalizer
    os.makedirs("models", exist_ok=True)
    model.save("models/aapl_recurrent_ppo")
    env.save("models/vec_normalize.pkl")
    print("Training Complete. Model saved to models/aapl_recurrent_ppo")

if __name__ == "__main__":
    train_agent()
