import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None
    spaces = None


class TradingEnvironment(gym.Env if gym is not None else object):

    metadata = {"render_modes": ["human"]}

    def __init__(self, frame, feature_columns, initial_cash=100000.0, transaction_cost_bps=5.0):

        self.frame = frame.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.transaction_cost_bps = transaction_cost_bps

        self.current_step = 0
        self.position = 0
        self.cash = initial_cash
        self.portfolio_value = initial_cash

        if spaces is not None:
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(feature_columns),),
                dtype=np.float32,
            )

    def reset(self, seed=None, options=None):

        self.current_step = 0
        self.position = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash

        observation = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}

        return observation, info

    def step(self, action):

        current_price = float(self.frame.iloc[self.current_step]["Close"])
        previous_value = self.portfolio_value

        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1

        next_step = min(self.current_step + 1, len(self.frame) - 1)
        next_price = float(self.frame.iloc[next_step]["Close"])
        price_return = 0.0 if current_price == 0 else (next_price / current_price) - 1.0
        transaction_cost = abs(action - 0) * (self.transaction_cost_bps / 10000.0)

        self.portfolio_value *= (1.0 + self.position * price_return - transaction_cost)
        reward = self.portfolio_value - previous_value
        self.current_step = next_step

        terminated = self.current_step >= len(self.frame) - 1
        truncated = False
        info = {"portfolio_value": self.portfolio_value}

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):

        row = self.frame.iloc[self.current_step]
        values = row[self.feature_columns].astype(float).fillna(0.0).to_numpy(dtype=np.float32)

        return values
