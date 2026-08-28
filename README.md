# TradingAgentCI

Deep reinforcement learning agents for financial trading environments, built on
[gym-anytrading](https://github.com/AminRezaei0x443/gym-anytrading) and Gymnasium.

Course project for the Deep Reinforcement Learning track: train and evaluate
DRL agents (A2C baseline first, expanding from there) on trading environments,
with a modular script layout so experiments stay reproducible.

## Structure

- `baseline_agent.py` — initial A2C agent on the default trading environment
- `scripts/` — modular training / evaluation scripts
- `data/` — datasets used by the environments
- `reports/` — documentation and result figures
- `Assignment/` — the original assignment specification

## Stack

- Python, PyTorch (via Stable-Baselines3 A2C)
- Gymnasium + gym-anytrading environments

## Status

Baseline stage: A2C agent training and evaluation running end to end.
Next: custom feature engineering on the observation space and a PPO comparison run.
