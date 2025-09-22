# 🧠 MarketMind

<div align="center">

![MarketMind Logo](https://img.shields.io/badge/MarketMind-AI%20Trading%20Bot-purple?style=for-the-badge&logo=bitcoin&logoColor=white)

[Live App](https://marketmind-sanskarmodi.streamlit.app/)

**Autonomous Bitcoin Trading Bot Powered by Deep Reinforcement Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg?style=flat&logo=mlflow)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Pipeline%20Versioning-blue.svg?style=flat&logo=dvc)](https://dvc.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg?style=flat&logo=streamlit)](https://streamlit.io)

</div>

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Training Pipeline](#-training-pipeline)
- [📊 Results](#-results)
- [🚀 Quick Start](#-quick-start)
- [🧪 Training Environment](#-training-environment)
- [📈 Performance Monitoring](#-performance-monitoring)
- [🌐 Web Interface](#-web-interface)
- [📁 Project Structure](#-project-structure)
- [🔬 Experiments & Tracking](#-experiments--tracking)
- [🤝 Contributing](#-contributing)
- [🔮 Future Enhancements](#-future-enhancements)
- [📜 License](#-license)

## 🌟 Overview

MarketMind is an advanced autonomous trading system that leverages **Deep Reinforcement Learning (DRL)** to make intelligent Bitcoin trading decisions. Built with the **Proximal Policy Optimization (PPO)** algorithm, it learns optimal trading strategies through continuous market interaction while managing risk and transaction costs.

### 🎯 What Makes MarketMind Special?

- **🤖 Intelligent Decision Making**: Uses PPO algorithm to learn complex trading patterns
- **📊 Technical Analysis**: Incorporates multiple technical indicators and market features
- **🔄 Long-Only Strategy**: Focuses on buy, hold, and sell decisions (no short selling)
- **📈 Real-time Predictions**: Live market data integration with Yahoo Finance
- **🌐 Interactive Dashboard**: Beautiful Streamlit web interface for monitoring and predictions
- **🔬 Experiment Tracking**: Comprehensive MLflow integration for model versioning and comparison
- **📈 Proven Outperformance**: RL agent achieved **~204% total return** vs **124% buy-and-hold** in backtests
- **🔄 Weekly Automatic Retraining**: Keep your model up-to-date with the latest market data using a single command (`python main.py`)

---

## ✨ Key Features

### 🧠 AI-Powered Trading Engine

- **Deep Reinforcement Learning** with PPO algorithm  
- **Custom Trading Environment** with realistic market conditions  
- **Feature Engineering** with technical indicators (SMA, volatility, volume analysis)  
- **Normalized Observations** for stable training  

### 📊 Comprehensive Pipeline

- **Automated Data Ingestion** from Yahoo Finance  
- **Feature Engineering** with technical indicators  
- **Model Training** with hyperparameter optimization  
- **Model Evaluation** with performance metrics and visualizations  
- **DVC Pipeline** for reproducible machine learning workflows  
- **Automatic Weekly Retraining** with `python main.py` for up-to-date models  

### 📈 Performance Analytics

- **Portfolio Performance Tracking**  
- **Risk-Adjusted Returns** analysis  
- **Action Distribution** visualization  
- **Comparison with Buy-and-Hold** strategy  
- **Real-time Performance Monitoring**  

### 🌐 User Interface

- **Interactive Web Dashboard** built with Streamlit  
- **Real-time Predictions**  
- **Historical Performance** charts and metrics  
- **Technical Indicators** visualization  
- **Model Status** monitoring  

---

## 🏗️ Training Pipeline

```mermaid
graph TD
    A[Yahoo Finance API] --> B[Data Ingestion]
    B --> C[Data Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Trading Environment]
    E --> F[PPO Agent Training]
    F --> G[Model Evaluation]
    G --> H[MLflow Logging]

    style A fill:#e1f5fe
    style F fill:#f3e5f5
    style H fill:#e8f5e8
````

### 🔄 Automatic Weekly Model Updates

MarketMind can **automatically retrain the model weekly** with the latest market data to ensure it stays relevant. This is done **with a single command**:

```bash
python main.py
```

This command:

* Pulls the latest market data
* Runs all pipeline stages (ingestion → preprocessing → training → evaluation)
* Updates the trained model artifacts
* Keeps the MLflow experiments up to date

---

## 📊 Results

### Backtest Performance (Pipeline Run)

| Metric                          | Value       |
| ------------------------------- | ----------- |
| **Initial Portfolio Value**     | 1.0         |
| **Final Portfolio Value**       | 3.0377      |
| **RL Agent Total Return (%)**   | **203.77%** |
| **Buy & Hold Total Return (%)** | 124.15%     |
| **Average Daily Rewards**       | 0.00247     |
| **Win Rate (%)**                | 32.03%      |

> ✅ **Outperformance:** RL Agent returned **\~204%** compared to **\~124%** for Buy & Hold on the same period, including transaction costs.

### Visual Comparison

* **RL Agent:** 🚀 203.8% total return
* **Buy & Hold:** 📈 124.1% total return

This demonstrates MarketMind’s ability to identify favorable trading opportunities beyond passive holding.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* pip or conda
* Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/sanskarmodi8/MarketMind.git
cd MarketMind
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install the package**

```bash
pip install -e .
```

### 🏃‍♂️ Running the Complete Pipeline

Ensure that `src/MarketMind/pipeline/stage_04_model_evaluation.py` is using `ModelEvaluation()` and not `ModelEvaluationMLFLOW()` if you don't have MLflow credentials in your `.env` file.

**Option 1: Run All Stages**

```bash
python main.py
```

**Option 2: Run Individual Stages**

```bash
# Data Ingestion
python src/MarketMind/pipeline/stage_01_data_ingestion.py

# Data Preprocessing
python src/MarketMind/pipeline/stage_02_data_preprocessing.py

# Model Training
python src/MarketMind/pipeline/stage_03_model_training.py

# Model Evaluation
python src/MarketMind/pipeline/stage_04_model_evaluation.py
```

**Option 3: Using DVC Pipeline Versioning**

```bash
dvc repro
```

This will retrain the model with the latest market data, update artifacts, and keep MLflow experiments current.

### 🌐 Launch Web Interface

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

---

## 🧪 Training Environment

### Action Space

* **0**: Hold (maintain current position)
* **1**: Buy (go long if not already positioned)
* **2**: Sell (close position if currently long)

### Observation Space

* **Window of Features**: Normalized technical indicators over time window
* **Position Flag**: Current position status (0 or 1)
* **Feature Engineering**:

  * Log returns
  * Short-term and long-term SMAs
  * Volatility measures
  * Price ratios and ranges
  * Volume z-scores

### Reward Function

* **Portfolio Growth**: Log change in portfolio value
* **Transaction Costs**: Realistic trading fees
* **Turnover Penalty**: Discourages excessive trading

---

## 📈 Performance Monitoring

### Key Metrics

* **Total Return**: Cumulative portfolio performance
* **Win Rate**: Percentage of profitable trades

### Visualizations

* Portfolio value over time
* Action distribution charts
* Performance comparison with benchmarks

---

## 📁 Project Structure

```
MarketMind/
│
├── 📊 artifacts/              # Generated artifacts and outputs
│   ├── data_ingestion/        # Raw market data
│   ├── data_preprocessing/    # Processed datasets
│   ├── model_training/        # Trained models and logs
│   ├── model_evaluation/      # Evaluation reports and plots
│   └── mlflow_runs/           # MLflow experiment artifacts
│
├── 📋 config/                 # Configuration files
│   └── config.yaml           # Base configuration
│
├── 📓 notebooks/             # Jupyter notebooks for exploration
│   ├── data_ingestion.ipynb
│   ├── data_preprocessing.ipynb
│   └── model_train_eval.ipynb
│
├── 🐍 src/MarketMind/        # Source code
│   ├── components/           # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   ├── config/               # Configuration management
│   ├── entity/               # Data classes and entities
│   ├── pipeline/             # Pipeline stages
│   └── utils/                # Utility functions
│
├── 🌐 app.py                 # Streamlit web application
├── 🚀 main.py                # Main pipeline runner
├── 📋 dvc.yaml               # DVC pipeline definition
├── ⚙️ params.yaml            # Hyperparameters and settings
├── 📦 requirements.txt       # Python dependencies
└── 📖 README.md              
```

### Environment Variables

Create a `.env` file for MLflow configuration:

```env
MLFLOW_TRACKING_URI=your-mlflow-uri
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-token
```

---

## 🔬 Experiments & Tracking

### MLflow Integration

* **Metrics Logging**: Performance metrics and hyperparameters
* **Artifact Storage**: Model files, plots, and reports

### DVC Pipeline

* **Reproducible Workflows**: Version-controlled pipeline stages
* **Dependency Management**: Automatic pipeline execution

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**

   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** if applicable
5. **Run the formatter**

   ```bash
   bash format.sh
   ```
6. **Commit your changes**

   ```bash
   git commit -m "Add amazing feature"
   ```
7. **Push to the branch**

   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request**

---

## 🔮 Future Enhancements

* [ ] **Multi-Asset Support**: Portfolio optimization across cryptocurrencies
* [ ] **Advanced Strategies**: Short selling and leverage support
* [ ] **Real-time Trading**: Live trading integration with exchanges

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⚠️ Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.

**Made with ❤️ by [Sanskar Modi](https://github.com/sanskarmodi8)**

⭐ **If you find this project helpful, please give it a star!** ⭐

</div>
``` README? (It looks very impressive to recruiters.)
