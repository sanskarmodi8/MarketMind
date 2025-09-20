import os
import pickle
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MarketMind import logger
from MarketMind.components.model_training import TradingEnv
from MarketMind.entity.config_entity import ModelEvaluationConfig
from MarketMind.utils.common import save_json

load_dotenv()  # load env vars for mlflow


# ========================================
# Model Evaluation Class
# ========================================
class ModelEvaluation:
    """
    Evaluates a trained RL agent on the test dataset and generates evaluation reports and plots.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Seeds for reproducibility
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

    # --- Helper Methods ---
    def _load_env(self, df):
        logger.info("Creating testing environment...")
        test_env = DummyVecEnv(
            [
                lambda: TradingEnv(
                    df=df,
                    window_size=self.config.window_size,
                    transaction_cost=self.config.transaction_cost,
                    initial_balance=self.config.initial_balance,
                    reward_scaling=self.config.reward_scaling,
                    deterministic=True,
                    turnover_penalty=self.config.turnover_penalty,
                )
            ]
        )
        logger.info("Loading VecNormalize statistics...")
        vec_normalize = VecNormalize.load(self.config.normalized_vec_env_path, test_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        return vec_normalize

    def _evaluate_agent(self, model, vec_normalize, df):
        logger.info("Starting agent evaluation...")
        obs = vec_normalize.reset()
        portfolio_values = [vec_normalize.envs[0].portfolio_value]
        actions_taken, rewards = [], []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = vec_normalize.step(action)
            reward = reward[0]
            done = done_vec[0]
            if done:
                break

            portfolio_values.append(vec_normalize.envs[0].portfolio_value)
            actions_taken.append(action)
            rewards.append(reward)

        return (
            np.array(portfolio_values),
            np.array(actions_taken),
            np.array(rewards),
        )

    def _save_plots_and_reports(self, df, portfolio_values, actions_taken, rewards):
        logger.info("Saving evaluation plots and reports...")

        # Portfolio Value over Time
        plt.figure(figsize=(12, 6))
        plt.plot(
            df.index[: len(portfolio_values)], portfolio_values, label="Portfolio Value"
        )
        plt.title("Portfolio Value over Time (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.savefig(os.path.join(self.config.plots_dir, "portfolio_vs_time.png"))
        plt.close()

        # Reward Distribution
        plt.figure(figsize=(8, 4))
        plt.hist(rewards, bins=50, alpha=0.7)
        plt.title("Reward Distribution on Test Set")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.config.plots_dir, "reward_distribution.png"))
        plt.close()

        # Agent Actions on Price
        prices = df["Close"].values[: len(actions_taken)]
        buy_points = np.where(actions_taken == 1)[0]
        sell_points = np.where(actions_taken == 2)[0]

        plt.figure(figsize=(14, 7))
        plt.plot(df.index[: len(prices)], prices, label="BTC Close Price", color="blue")
        plt.scatter(
            df.index[buy_points],
            prices[buy_points],
            marker="^",
            color="green",
            label="Buy",
            alpha=0.8,
            s=100,
        )
        plt.scatter(
            df.index[sell_points],
            prices[sell_points],
            marker="v",
            color="red",
            label="Sell",
            alpha=0.8,
            s=100,
        )
        plt.title("Agent Actions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plots_dir, "agent_actions.png"))
        plt.close()

        # Agent vs Buy & Hold
        initial_cash = portfolio_values[0]
        btc_prices = df["Close"].values[: len(portfolio_values)]
        buy_and_hold_equity = initial_cash * (btc_prices / btc_prices[0])

        plt.figure(figsize=(12, 6))
        plt.plot(
            df.index[: len(portfolio_values)],
            portfolio_values,
            label="RL Agent Portfolio",
            color="blue",
        )
        plt.plot(
            df.index[: len(portfolio_values)],
            buy_and_hold_equity,
            label="Buy & Hold BTC",
            color="orange",
            linestyle="--",
        )
        plt.title("Agent vs Buy & Hold (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.plots_dir, "agent_vs_buy_and_hold.png"))
        plt.close()

        # Summary Metrics
        agent_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        bh_return = (buy_and_hold_equity[-1] / buy_and_hold_equity[0] - 1) * 100

        report = {
            "Initial Portfolio Value": float(portfolio_values[0]),
            "Final Portfolio Value": float(portfolio_values[-1]),
            "Total Return (%)": agent_return,
            "Avg Daily Rewards": float(np.mean(rewards)),
            "Win Rate (%)": np.mean(rewards > 0) * 100,
        }
        save_json(
            Path(os.path.join(self.config.report_dir), "evaluation_report.json"), report
        )

        report2 = {
            "RL Agent Total Return (%)": agent_return,
            "Buy & Hold Total Return (%)": bh_return,
        }
        save_json(
            Path(os.path.join(self.config.report_dir), "comparison_with_baseline.json"),
            report2,
        )

    # --- Public Run Method ---
    def run(self):
        """
        Executes the model evaluation pipeline:
        - Loads preprocessed test data
        - Loads the trained PPO model and VecNormalize stats
        - Evaluates the model on the test environment
        - Saves plots and evaluation reports
        """
        logger.info("Reading preprocessed test data...")
        df = pd.read_csv(
            self.config.preprocessed_test_data_path, index_col=0, parse_dates=True
        )

        vec_normalize = self._load_env(df)
        model_path = os.path.join(self.config.model_dir, "best_model.zip")
        logger.info(f"Loading trained PPO model from {model_path}...")
        model = PPO.load(model_path)

        portfolio_values, actions_taken, rewards = self._evaluate_agent(
            model, vec_normalize, df
        )
        self._save_plots_and_reports(df, portfolio_values, actions_taken, rewards)

        logger.info("Model evaluation completed successfully.")


# ========================================
# Model Evaluation Class with MLflow
# ========================================
class ModelEvaluationMLFLOW:
    """
    Evaluates a trained RL agent on the test dataset and logs everything to MLflow.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Seeds for reproducibility
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

    # --- Helper Methods ---
    def _load_env(self, df):
        logger.info("Creating testing environment...")
        test_env = DummyVecEnv(
            [
                lambda: TradingEnv(
                    df=df,
                    window_size=self.config.window_size,
                    transaction_cost=self.config.transaction_cost,
                    initial_balance=self.config.initial_balance,
                    reward_scaling=self.config.reward_scaling,
                    deterministic=True,
                    turnover_penalty=self.config.turnover_penalty,
                )
            ]
        )
        logger.info("Loading VecNormalize statistics...")
        vec_normalize = VecNormalize.load(self.config.normalized_vec_env_path, test_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        return vec_normalize

    def _evaluate_agent(self, model, vec_normalize, df):
        logger.info("Starting agent evaluation...")
        obs = vec_normalize.reset()
        portfolio_values = [vec_normalize.envs[0].portfolio_value]
        actions_taken, rewards = [], []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = vec_normalize.step(action)
            reward = reward[0]
            done = done_vec[0]
            if done:
                break

            portfolio_values.append(vec_normalize.envs[0].portfolio_value)
            actions_taken.append(action)
            rewards.append(reward)

        return (
            np.array(portfolio_values),
            np.array(actions_taken),
            np.array(rewards),
        )

    def _save_plots_and_reports(self, df, portfolio_values, actions_taken, rewards):
        logger.info("Saving evaluation plots and reports...")

        # Create local dirs
        Path(self.config.plots_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.report_dir).mkdir(parents=True, exist_ok=True)

        # Portfolio Value over Time
        portfolio_plot = os.path.join(self.config.plots_dir, "portfolio_vs_time.png")
        plt.figure(figsize=(12, 6))
        plt.plot(
            df.index[: len(portfolio_values)], portfolio_values, label="Portfolio Value"
        )
        plt.title("Portfolio Value over Time (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.savefig(portfolio_plot)
        plt.close()

        # Reward Distribution
        reward_plot = os.path.join(self.config.plots_dir, "reward_distribution.png")
        plt.figure(figsize=(8, 4))
        plt.hist(rewards, bins=50, alpha=0.7)
        plt.title("Reward Distribution on Test Set")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.savefig(reward_plot)
        plt.close()

        # Agent Actions on Price
        prices = df["Close"].values[: len(actions_taken)]
        buy_points = np.where(actions_taken == 1)[0]
        sell_points = np.where(actions_taken == 2)[0]
        actions_plot = os.path.join(self.config.plots_dir, "agent_actions.png")

        plt.figure(figsize=(14, 7))
        plt.plot(df.index[: len(prices)], prices, label="BTC Close Price", color="blue")
        plt.scatter(
            df.index[buy_points],
            prices[buy_points],
            marker="^",
            color="green",
            label="Buy",
            alpha=0.8,
            s=100,
        )
        plt.scatter(
            df.index[sell_points],
            prices[sell_points],
            marker="v",
            color="red",
            label="Sell",
            alpha=0.8,
            s=100,
        )
        plt.title("Agent Actions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(actions_plot)
        plt.close()

        # Agent vs Buy & Hold
        agent_vs_bh_plot = os.path.join(
            self.config.plots_dir, "agent_vs_buy_and_hold.png"
        )
        initial_cash = portfolio_values[0]
        btc_prices = df["Close"].values[: len(portfolio_values)]
        buy_and_hold_equity = initial_cash * (btc_prices / btc_prices[0])

        plt.figure(figsize=(12, 6))
        plt.plot(
            df.index[: len(portfolio_values)],
            portfolio_values,
            label="RL Agent Portfolio",
            color="blue",
        )
        plt.plot(
            df.index[: len(portfolio_values)],
            buy_and_hold_equity,
            label="Buy & Hold BTC",
            color="orange",
            linestyle="--",
        )
        plt.title("Agent vs Buy & Hold (Test Set)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(agent_vs_bh_plot)
        plt.close()

        # Summary Metrics
        agent_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        bh_return = (buy_and_hold_equity[-1] / buy_and_hold_equity[0] - 1) * 100

        report = {
            "Initial Portfolio Value": float(portfolio_values[0]),
            "Final Portfolio Value": float(portfolio_values[-1]),
            "Total Return (%)": agent_return,
            "Avg Daily Rewards": float(np.mean(rewards)),
            "Win Rate (%)": np.mean(rewards > 0) * 100,
        }
        save_json(
            Path(os.path.join(self.config.report_dir), "evaluation_report.json"), report
        )

        report2 = {
            "RL Agent Total Return (%)": agent_return,
            "Buy & Hold Total Return (%)": bh_return,
        }
        save_json(
            Path(os.path.join(self.config.report_dir), "comparison_with_baseline.json"),
            report2,
        )

        return (
            portfolio_plot,
            reward_plot,
            actions_plot,
            agent_vs_bh_plot,
            report,
            report2,
        )

    # --- Public Run Method ---
    def run(self):
        """
        Executes the model evaluation pipeline and logs to MLflow.
        """
        logger.info("Reading preprocessed test data...")
        df = pd.read_csv(
            self.config.preprocessed_test_data_path, index_col=0, parse_dates=True
        ).sort_index()

        # Start MLflow experiment
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("MarketMind_Model_Evaluation")
        with mlflow.start_run(run_name="model_evaluation"):
            vec_normalize = self._load_env(df)
            model_path = os.path.join(self.config.model_dir, "best_model.zip")
            logger.info(f"Loading trained PPO model from {model_path}...")
            model = PPO.load(model_path)

            portfolio_values, actions_taken, rewards = self._evaluate_agent(
                model, vec_normalize, df
            )

            (
                portfolio_plot,
                reward_plot,
                actions_plot,
                agent_vs_bh_plot,
                report,
                report2,
            ) = self._save_plots_and_reports(
                df, portfolio_values, actions_taken, rewards
            )

            logger.info("Logging metrics and artifacts to MLflow...")

            # Log metrics
            mlflow.log_metric("total_return_pct", float(report["Total Return (%)"]))
            mlflow.log_metric("avg_daily_rewards", float(report["Avg Daily Rewards"]))
            mlflow.log_metric("win_Rate_pct", float(report["Win Rate (%)"]))
            mlflow.log_metric(
                "buy_and_hold_return_pct", float(report2["Buy & Hold Total Return (%)"])
            )

            # Log plots as artifacts
            mlflow.log_artifact(portfolio_plot, artifact_path="plots")
            mlflow.log_artifact(reward_plot, artifact_path="plots")
            mlflow.log_artifact(actions_plot, artifact_path="plots")
            mlflow.log_artifact(agent_vs_bh_plot, artifact_path="plots")

            # Log reports as artifacts
            mlflow.log_artifact(
                os.path.join(self.config.report_dir, "evaluation_report.json"),
                artifact_path="reports",
            )
            mlflow.log_artifact(
                os.path.join(self.config.report_dir, "comparison_with_baseline.json"),
                artifact_path="reports",
            )

            # Save VecNormalize as artifact
            vecnorm_file = os.path.join(self.config.root_dir, "vecnormalize.pkl")
            with open(vecnorm_file, "wb") as f:
                pickle.dump(vec_normalize, f)
            mlflow.log_artifact(vecnorm_file, artifact_path="artifacts")

            # Log model (Stable Baselines model file itself)
            mlflow.log_artifact(model_path, artifact_path="artifacts")

            logger.info("Model evaluation completed successfully and logged to MLflow.")
