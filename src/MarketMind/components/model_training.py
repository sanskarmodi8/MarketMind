import os
import random
import multiprocessing

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from MarketMind import logger
from MarketMind.entity.config_entity import ModelTrainingConfig

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Torch GPU performance tweaks
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# -------------------------
# Custom EvalCallback to save VecNormalize
# -------------------------
class SaveVecNormalizeCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)

    def _on_step(self) -> bool:
        result = super()._on_step()
        if (
            self.best_model_save_path is not None
            and self.n_calls % self.eval_freq == 0
        ):
            vecnormalize_path = os.path.join(
                self.best_model_save_path, "vecnormalize.pkl"
            )
            self.model.get_env().save(vecnormalize_path)
        return result


# =========================
# Trading Environment
# =========================
class TradingEnv(gym.Env):
    """
    Long-only Trading Environment.
    Actions:
        0 = Hold
        1 = Buy (go long)
        2 = Sell (close position)
    """

    def __init__(
        self,
        df,
        feature_cols=None,
        window_size=100,
        transaction_cost=0.001,
        initial_balance=1.0,
        reward_scaling=1.0,
        deterministic=True,
        turnover_penalty=0.0005,
    ):
        super().__init__()
        self.df = df.copy()
        self.feature_cols = feature_cols or [
            "log_return",
            "SMA_short",
            "SMA_long",
            "volatility",
            "open_close_ratio",
            "high_low_range",
            "volume_zscore",
        ]
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling
        self.deterministic = deterministic
        self.turnover_penalty = turnover_penalty

        self.start_index = self.window_size
        self.end_index = len(self.df) - 1

        self.action_space = spaces.Discrete(3)
        self.n_features = len(self.feature_cols)
        obs_len = self.window_size * self.n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = (
            self.start_index
            if self.deterministic
            else np.random.randint(self.start_index, self.end_index - 1)
        )
        self.position = 0
        self.cash = self.initial_balance
        self.invested = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.cash
        self.trades = []
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        window = self.df.iloc[start:end][self.feature_cols].values

        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-9
        norm_window = (window - mean) / std

        obs = np.concatenate(
            [norm_window.flatten(), np.array([self.position], dtype=np.float32)]
        )
        return obs.astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        info = {}

        prev_portfolio_value = self.portfolio_value
        price_now = float(self.df["Close"].iloc[self.current_step])
        price_next = float(self.df["Close"].iloc[self.current_step + 1])
        prev_position = self.position

        if action == 1 and self.position == 0:  # Buy
            trade_notional = self.cash
            cost = self.transaction_cost * trade_notional
            self.invested = self.cash - cost
            self.entry_price = price_now
            self.position = 1
            self.cash = 0.0
            self.trades.append((self.current_step, "BUY", price_now, self.portfolio_value))

        elif action == 2 and self.position == 1:  # Sell
            current_value = self.invested * (price_now / self.entry_price)
            cost = self.transaction_cost * current_value
            self.cash = current_value - cost
            self.invested = 0.0
            self.entry_price = 0.0
            self.position = 0
            self.trades.append((self.current_step, "SELL", price_now, self.cash))

        if self.position == 1:
            asset_value_next = self.invested * (price_next / self.entry_price)
            self.portfolio_value = asset_value_next
        else:
            self.portfolio_value = self.cash

        reward = np.log(self.portfolio_value / max(prev_portfolio_value, 1e-9))
        turnover = abs(self.position - prev_position)
        reward -= self.turnover_penalty * turnover
        reward *= self.reward_scaling

        self.current_step += 1
        if self.current_step >= self.end_index:
            done = True

        obs = self._get_obs()
        return obs, float(reward), done, False, info

    def close(self):
        pass


# =========================
# Model Training
# =========================
class ModelTraining:
    """
    Class to handle model training using PPO algorithm from Stable Baselines3.
    """

    def __init__(self, config: ModelTrainingConfig, n_envs: int = None):
        self.config = config
        self.n_envs = n_envs or multiprocessing.cpu_count() // 2  # Use half cores by default

    def run(self):
        logger.info("Loading preprocessed training data...")
        df = pd.read_csv(
            self.config.preprocessed_train_data_path, index_col=0, parse_dates=True
        ).sort_index()

        logger.info(f"Initializing {self.n_envs} parallel training environments...")

        def make_env(deterministic=False):
            return lambda: TradingEnv(
                df,
                window_size=self.config.window_size,
                transaction_cost=self.config.transaction_cost,
                initial_balance=self.config.initial_balance,
                reward_scaling=self.config.reward_scaling,
                deterministic=deterministic,
                turnover_penalty=self.config.turnover_penalty,
            )

        # Parallel training envs
        train_env = SubprocVecEnv([make_env(False) for _ in range(self.n_envs)])
        eval_env = SubprocVecEnv([make_env(True) for _ in range(1)])

        vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_eval_env.training = False
        vec_eval_env.norm_reward = False

        logger.info("Configuring PPO model...")
        policy_kwargs = dict(net_arch=self.config.net_arch, activation_fn=torch.nn.ReLU)
        lr_schedule = lambda p: self.config.initial_lr * p

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr_schedule,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tb_log_dir,
            device="cuda", 
        )

        eval_callback = SaveVecNormalizeCallback(
            vec_eval_env,
            best_model_save_path=self.config.model_dir,
            log_path=self.config.model_dir,
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False,
        )

        logger.info("Starting training with full compute power...")
        model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=eval_callback,
            tb_log_name="PPO",
        )
        logger.info("Training complete.")
