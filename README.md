# DRL Trading Agent Assignment

Project based on `gym-anytrading` for the Deep Reinforcement Learning course (8th Semester).

## Project Structure
- `Assignment/`: Contains the original assignment PDF and its text version.
- `data/`: Placeholder for custom datasets.
- `models/`: Saved trained agents.
- `reports/`: Documentations and figures.
- `scripts/`: Modular Python scripts.
- `baseline_agent.py`: Initial test script using A2C.

## Requirements
- Gymnasium
- gym-anytrading
- Stable-Baselines3
- QuantStats
- Matplotlib, Pandas, Numpy

## Setup
The environment is setup in the `venv` folder. To use it:
```bash
source venv/bin/activate
```

## First Steps
1. Run `python baseline_agent.py` to verify the installation and see a basic agent in action.
2. Experiment with different algorithms (PPO, DQN).
3. Add technical indicators (SMA, RSI) to the observation space to improve performance.
4. Compare performance between Stocks and Forex environments.
