import os
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from MarketMind import logger
from MarketMind.components.model_training import TradingEnv as TradingEnvPredict

# constants
LOCAL_MLFLOW_ARTIFACT_DIR = Path("artifacts") / "mlflow_runs"
LOCAL_MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MLFLOW_TRACKING_URI = "https://dagshub.com/sanskarmodi8/MarketMind.mlflow"
EXPERIMENT_NAME = "MarketMind_Model_Evaluation"
TRAINING_WINDOW_SIZE = 100  # must match training config


class PredictionPipeline:
    def __init__(self):
        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self.feature_cols = [
            "log_return",
            "SMA_short",
            "SMA_long",
            "volatility",
            "open_close_ratio",
            "high_low_range",
            "volume_zscore",
        ]

        # Load .env and set mlflow tracking uri (env var MLFLOW_TRACKING_URI optional)
        load_dotenv()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            # set_tracking_uri can raise if invalid; still proceed and log
            logger.warning(
                f"Could not set mlflow tracking uri to {tracking_uri} (continuing)."
            )
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    def _latest_run_artifact_paths(
        self,
    ) -> (Optional[Path], Optional[Path], Optional[str]):
        """
        Find latest run for the experiment and return local artifact paths:
        (model_path, vecnormalize_path, run_id) or (None, None, None) on failure.
        """
        try:
            exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if exp is None:
                logger.error(f"MLflow experiment '{EXPERIMENT_NAME}' not found.")
                return None, None, None

            # search_runs sorted by start_time desc
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs is None or runs.empty:
                logger.error("No runs found in experiment.")
                return None, None, None

            latest_run_id = runs.iloc[0].run_id
            logger.info(f"Latest MLflow run id: {latest_run_id}")

            # local destination for artifacts
            dest = LOCAL_MLFLOW_ARTIFACT_DIR / latest_run_id
            if dest.exists():
                logger.info(
                    f"Artifacts for run {latest_run_id} already downloaded at {dest}"
                )
            else:
                logger.info(
                    f"Downloading artifacts for run {latest_run_id} into {dest} ..."
                )
                dest.mkdir(parents=True, exist_ok=True)
                # download artifacts from run root (artifact_path empty downloads root artifacts)
                # some MLflow servers require artifact_path argument; here we try root then fallback to 'artifacts'
                try:
                    local_root = mlflow.artifacts.download_artifacts(
                        run_id=latest_run_id, artifact_path=""
                    )
                except Exception:
                    local_root = mlflow.artifacts.download_artifacts(
                        run_id=latest_run_id, artifact_path="artifacts"
                    )
                # Move contents into dest if download returns different folder
                local_root_path = Path(local_root)
                # ensure dest contains files
                if local_root_path.exists():
                    # if local_root_path is a file (unlikely), copy file into dest
                    if local_root_path.is_file():
                        (dest / local_root_path.name).write_bytes(
                            local_root_path.read_bytes()
                        )
                    else:
                        # copy all files/dirs
                        for child in local_root_path.iterdir():
                            target = dest / child.name
                            if child.is_dir():
                                # copy tree (simple)
                                import shutil

                                if target.exists():
                                    shutil.rmtree(target)
                                shutil.copytree(child, target)
                            else:
                                target.write_bytes(child.read_bytes())
                logger.info(f"Downloaded artifacts to {dest}")

            # look for expected artifact names under dest
            model_file = dest / "artifacts" / "best_model.zip"
            vec_file = dest / "artifacts" / "vecnormalize.pkl"
            # tolerate alternate locations: dest/best_model.zip etc
            if not model_file.exists():
                alt = dest / "best_model.zip"
                if alt.exists():
                    model_file = alt
            if not vec_file.exists():
                alt = dest / "vecnormalize.pkl"
                if alt.exists():
                    vec_file = alt

            if not model_file.exists() or not vec_file.exists():
                logger.error(
                    f"Expected artifacts not found in {dest} (model: {model_file.exists()}, vec: {vec_file.exists()})"
                )
                return None, None, latest_run_id

            return model_file, vec_file, latest_run_id

        except Exception as e:
            logger.error(f"Error while fetching latest run artifacts: {e}")
            return None, None, None

    def load_model(self) -> bool:
        """
        Loads model + VecNormalize. Downloads artifacts from MLflow only once (per run).
        Returns True on success.
        """
        try:
            if self.model is not None and self.vec_normalize is not None:
                logger.info("Model already loaded in memory.")
                return True

            model_path, vecnormalize_path, run_id = self._latest_run_artifact_paths()
            if model_path is None or vecnormalize_path is None:
                logger.error("Could not find model or vecnormalize artifact.")
                return False

            # create dummy env matching training env (window size) to load VecNormalize
            dummy_rows = max(200, TRAINING_WINDOW_SIZE * 2)
            dummy_df = pd.DataFrame(
                {
                    "Close": [100.0] * dummy_rows,
                    **{col: [0.0] * dummy_rows for col in self.feature_cols},
                }
            )
            dummy_env = DummyVecEnv(
                [
                    lambda: TradingEnvPredict(
                        dummy_df, self.feature_cols, window_size=TRAINING_WINDOW_SIZE
                    )
                ]
            )

            logger.info(f"Loading PPO model from {model_path} ...")
            self.model = PPO.load(str(model_path))
            logger.info("Model loaded. Loading VecNormalize...")

            self.vec_normalize = VecNormalize.load(str(vecnormalize_path), dummy_env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False

            logger.info("Model and VecNormalize loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model/vecnormalize: {e}")
            return False

    def fetch_live_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch live data from Yahoo (simple wrapper)."""
        try:
            logger.info(f"Fetching {days} days of data for {symbol} ...")
            ticker = yf.Ticker(symbol)
            if isinstance(days, int):
                period = f"{days}d"
            else:
                period = str(days)
            data = ticker.history(period=period)
            if data is None or data.empty:
                logger.warning("No data returned from yf")
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features copy of your preprocessing pipeline (safe and idempotent)."""
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
            logger.error(f"Preprocessing error: {e}")
            return pd.DataFrame()

    def get_prediction(self, symbol: str, days: int = 365) -> Dict:
        """
        Returns a dict with keys:
            success (bool), symbol, prediction (BUY/SELL/HOLD) if success, current_price, price_change_24h,
            technical_indicators dict, timestamp, data_points_used
        """
        try:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Model load failed",
                    "symbol": symbol,
                }

            raw = self.fetch_live_data(symbol, days)
            if raw is None or raw.empty:
                return {
                    "success": False,
                    "error": "No data available",
                    "symbol": symbol,
                }

            processed = self.preprocess_data(raw)
            if processed is None or processed.empty:
                return {
                    "success": False,
                    "error": "Preprocessing failed or not enough data",
                    "symbol": symbol,
                }

            if len(processed) < TRAINING_WINDOW_SIZE:
                return {
                    "success": False,
                    "error": f"Not enough historical points (need >= {TRAINING_WINDOW_SIZE})",
                    "symbol": symbol,
                }

            env = TradingEnvPredict(
                processed, self.feature_cols, window_size=TRAINING_WINDOW_SIZE
            )
            obs, _ = env.reset()
            obs = obs.reshape(1, -1)

            try:
                obs_norm = (
                    self.vec_normalize.normalize_obs(obs)
                    if self.vec_normalize is not None
                    else obs
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"VecNormalize fail: {e}",
                    "symbol": symbol,
                }

            action, _ = self.model.predict(obs_norm, deterministic=True)
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            prediction = action_map.get(int(action[0]), "HOLD")

            current_price = float(processed["Close"].iloc[-1])
            prev_price = (
                float(processed["Close"].iloc[-2])
                if len(processed) > 1
                else current_price
            )
            price_change_24h = (
                (current_price / prev_price - 1) * 100.0 if prev_price != 0 else 0.0
            )

            sma_short = float(processed["SMA_short"].iloc[-1])
            sma_long = float(processed["SMA_long"].iloc[-1])
            volatility = float(processed["volatility"].iloc[-1])

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
                "timestamp": processed.index[-1].isoformat(),
                "data_points_used": len(processed),
            }

        except Exception as e:
            logger.error(f"Prediction pipeline error: {e}")
            return {"success": False, "error": str(e), "symbol": symbol}


# singleton instance (safe to import)
prediction_pipeline = PredictionPipeline()
