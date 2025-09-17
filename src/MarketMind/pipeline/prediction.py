# prediction.py (cleaned version)

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MarketMind import logger
from MarketMind.components.model_training import TradingEnv as TradingEnvPredict


class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.vec_normalize = None
        self.feature_cols = [
            "log_return",
            "SMA_short",
            "SMA_long",
            "volatility",
            "open_close_ratio",
            "high_low_range",
            "volume_zscore",
        ]

    def load_model(self) -> bool:
        try:
            model_path = "notebooks/models/best_model.zip"
            vec_normalize_path = "notebooks/vec_normalize.pkl"

            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return False
            if not os.path.exists(vec_normalize_path):
                logger.error(f"VecNormalize stats not found at {vec_normalize_path}")
                return False

            dummy_df = pd.DataFrame(
                {
                    "Close": [100] * 200,
                    **{col: [0.0] * 200 for col in self.feature_cols},
                }
            )
            dummy_env = DummyVecEnv(
                [lambda: TradingEnvPredict(dummy_df, self.feature_cols)]
            )

            self.model = PPO.load(model_path)
            self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            logger.info("Model + normalization loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def fetch_live_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        try:
            logger.info(f"Fetching live data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", auto_adjust=False)
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
            df["SMA_short"] = df["Close"].rolling(window=20).mean()
            df["SMA_long"] = df["Close"].rolling(window=50).mean()
            df["volatility"] = df["log_return"].rolling(window=20).std()
            df["open_close_ratio"] = df["Open"] / df["Close"]
            df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
            vol_mean = df["Volume"].rolling(20).mean()
            vol_std = df["Volume"].rolling(20).std()
            df["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return pd.DataFrame()

    def get_prediction(self, symbol: str, days: int = 365) -> Dict:
        try:
            if self.model is None or self.vec_normalize is None:
                if not self.load_model():
                    return {"success": False, "error": "Failed to load model", "symbol": symbol}

            raw_data = self.fetch_live_data(symbol, days)
            if raw_data is None or raw_data.empty:
                return {"success": False, "error": "No data available", "symbol": symbol}

            processed_data = self.preprocess_data(raw_data)
            if processed_data.empty:
                return {"success": False, "error": "Preprocessing failed", "symbol": symbol}

            # Safety check: ensure we have enough rows
            if len(processed_data) < 2:
                return {"success": False, "error": "Not enough data after preprocessing", "symbol": symbol}

            env = TradingEnvPredict(processed_data, self.feature_cols)

            obs, _ = env.reset()
            obs = obs.reshape(1, -1)

            # Safety: check VecNormalize
            if self.vec_normalize is None:
                obs_normalized = obs
            else:
                try:
                    obs_normalized = self.vec_normalize.normalize_obs(obs)
                except Exception as e:
                    return {"success": False, "error": f"VecNormalize failed: {e}", "symbol": symbol}

            # Make prediction
            action, _ = self.model.predict(obs_normalized, deterministic=True)
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            prediction = action_map.get(int(action[0]), "HOLD")

            # Compute metrics safely
            current_price = float(processed_data["Close"].iloc[-1])
            price_change_24h = (
                float(processed_data["Close"].iloc[-1] / processed_data["Close"].iloc[-2] - 1) * 100
            )
            sma_short = float(processed_data["SMA_short"].iloc[-1])
            sma_long = float(processed_data["SMA_long"].iloc[-1])
            volatility = float(processed_data["volatility"].iloc[-1])

            return {
                "success": True,
                "symbol": symbol,
                "prediction": prediction,
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "technical_indicators": {
                    "sma_short": sma_short,
                    "sma_long": sma_long,
                    "volatility": volatility,
                    "trend": "BULLISH" if sma_short > sma_long else "BEARISH",
                },
                "timestamp": processed_data.index[-1].isoformat(),
                "data_points_used": len(processed_data),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "symbol": symbol}


    def get_portfolio_recommendations(
        self, symbols: List[str], days: int = 365
    ) -> Dict:
        recommendations = {}
        for symbol in symbols:
            recommendations[symbol] = self.get_prediction(symbol, days)
        return {"success": True, "recommendations": recommendations}


prediction_pipeline = PredictionPipeline()
