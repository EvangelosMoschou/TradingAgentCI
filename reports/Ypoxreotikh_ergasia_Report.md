# Ypoxreotikh Ergasia: Deep Reinforcement Learning for Trading

**Course:** Computational Intelligence - Deep Reinforcement Learning
**Environment:** gym-anytrading (Stocks)

## 1. Problem Description
The problem addressed in this assignment is the development of an autonomous trading agent that learns to optimize profit by trading sequentially in the stock market. The agent operates within the `gym-anytrading` environment. At each time step, it observes a window of past market states and executes one of two actions: `Hold/Short (0)` or `Buy/Long (1)`. The agent's goal is to maximize the total profit achieved over the trading period.

To ensure the agent has sufficient information to make optimal decisions, the raw historical stock data (Open, High, Low, Close) from Apple Inc. (`AAPL`) is enriched with standard technical indicators:
- Simple Moving Averages (SMA 20, SMA 50)
- Relative Strength Index (RSI 14)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands (Length 20)

## 2. Neural Network and DRL Algorithm
The agent was trained using **Recurrent Proximal Policy Optimization (RecurrentPPO)**, an Actor-Critic algorithm that natively supports Recurrent Neural Networks (RNNs).

Since financial time-series data contains significant temporal dependencies, standard feed-forward networks (MLPs) struggle to recognize sequential patterns. To satisfy the assignment's requirement and improve performance, a **Long Short-Term Memory (LSTM)** network (a type of RNN) was used as the policy's feature extractor. The LSTM sequentially processes the observation window (e.g., 20 previous days) and retains a hidden state memory, allowing the Agent to capture long-term market trends before the final Multi-Layer Perceptron (MLP) outputs the action probabilities (Actor) and state-value estimations (Critic).

**Algorithm:** RecurrentPPO (from `sb3-contrib`)
**Policy Architecture:** `MlpLstmPolicy`
**Environment Wrapper:** `VecNormalize` (For normalizing observations and rewards, greatly enhancing LSTM stability)

## 3. Performance Metrics and Hyperparameters
A systematic experimentation grid was conducted to observe the algorithm's performance under various hyperparameters, specifically the Learning Rate and the Observation Window Size.

**Evaluation Setup:** 
- The agent was trained on the first ~80% of the data (from 2010 to late 2021). 
- It was evaluated chronologically on the final ~20% of the unseen data (Testing Dataset).

### Experiment Results Table
The following table summarizes the Profit obtained on the **Test Set** by varying the learning rate and window size. Profit is expressed as a multiplier (e.g., 1.5 equates to ending the period with 1.5x the starting capital).

| Learning Rate | Window Size | Total Profit | Total Reward (Internal metric) |
| ------------- | ----------- | ------------ | ------------------------------ |
| To be Added   | To be Added | To be Added  | To be Added                    |

*(The corresponding action charts (Buy/Sell annotations over the price graph) are located in the `reports/figures/` folder).*

## 4. Execution Instructions
To execute the experiments and replicate these results:

1. **Setup Environment:** Provide Python 3.12 and create a virtual environment, then install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install gymnasium stable-baselines3[extra] sb3-contrib gym-anytrading yfinance pandas-ta matplotlib quantstats
   ```
2. **Download and Process Data:**
   Run the data preparation script to download AAPL data and calculate indicators.
   ```bash
   python scripts/data_processing.py
   ```
3. **Train the Baseline LSTM Agent:**
   ```bash
   python scripts/train.py
   ```
4. **Evaluate the Baseline Agent:**
   ```bash
   python scripts/evaluate.py
   ```
5. **Run the Hyperparameter Experiments:**
   This will train multiple agents across the hyperparameter grid and save the profit plots and CSV table to `reports/`.
   ```bash
   python scripts/experiment.py
   ```
