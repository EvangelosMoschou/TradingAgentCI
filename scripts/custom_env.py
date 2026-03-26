import numpy as np
from gym_anytrading.envs import StocksEnv

class CustomStocksEnv(StocksEnv):
    """
    A custom trading environment that extends StocksEnv.
    It passes the custom technical indicators to the agent's observation space.
    """
    def __init__(self, df, window_size, frame_bound, **kwargs):
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound, **kwargs)

    def _process_data(self):
        """
        Extracts prices and features from the DataFrame.
        This overrides the default _process_data to include ALL our technical indicators,
        rather than just the OHLC data and their diffs.
        """
        # Prices are typically the 'Close' prices used to calculate rewards/profits
        prices = self.df.loc[:, 'Close'].to_numpy()
        
        # Select all relevant feature columns. 
        # We drop non-feature columns if any exist.
        drop_cols = ['Dividends', 'Stock Splits']
        cols = [c for c in self.df.columns if c not in drop_cols]
        
        # We use these columns as our signal features. 
        # Note: Non-stationary data (like raw prices) can be hard for neural networks.
        # A common practice is to calculate percentage returns or differences,
        # but Stable-Baselines3's VecNormalize wrapper handles standard normalization during training.
        signal_features = self.df.loc[:, cols].to_numpy()
        
        return prices, signal_features
