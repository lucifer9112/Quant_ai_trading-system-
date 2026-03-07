class StableBaselinesTradingAgent:

    def __init__(self, algorithm="PPO", policy="MlpPolicy", **model_kwargs):

        self.algorithm = algorithm
        self.policy = policy
        self.model_kwargs = model_kwargs
        self.model = None

    def _algorithm_class(self):

        try:
            import stable_baselines3 as sb3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "stable-baselines3 is not installed. Install it with `pip install stable-baselines3 gymnasium`."
            ) from exc

        if not hasattr(sb3, self.algorithm):
            raise ValueError(f"Unsupported Stable-Baselines3 algorithm: {self.algorithm}")

        return getattr(sb3, self.algorithm)

    def train(self, env, total_timesteps=10000):

        algorithm_class = self._algorithm_class()
        self.model = algorithm_class(self.policy, env, verbose=0, **self.model_kwargs)
        self.model.learn(total_timesteps=total_timesteps)

        return self.model

    def predict(self, observation, deterministic=True):

        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        action, _ = self.model.predict(observation, deterministic=deterministic)

        return action
