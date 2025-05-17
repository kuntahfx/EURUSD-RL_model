# EURUSD-RL_model creation

# ForexPro Trading System Documentation

**created by Kuntahfx**
contact; **kuntahkays@gmail.com**

## Overview
The **ForexPro Trading System** is a sophisticated, end-to-end solution for developing, training, and evaluating high-frequency forex trading strategies using reinforcement learning (RL). It integrates synthetic data generation, advanced feature engineering, and a custom Gym environment to train a trading agent optimized for scalping on EUR/USD and GBP/USD currency pairs. The system leverages state-of-the-art RL algorithms (TQC from `sb3-contrib`), feature selection, and curriculum learning to achieve robust performance in simulated market conditions. it is meant to be integrated with MQL5 if it all goes well

This documentation provides a comprehensive guide to the system's architecture, implementation details, and integration instructions for Python developers. It is designed to enable programmers to understand, extend, and deploy the system effectively.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Components](#components)
   - [Synthetic Data Generation](#synthetic-data-generation)
   - [Feature Engineering](#feature-engineering)
   - [Reinforcement Learning Environment](#reinforcement-learning-environment)
3. [Installation and Setup](#installation-and-setup)
4. [Integration in Python](#integration-in-python)
   - [Running the Pipeline](#running-the-pipeline)
   - [Extending the System](#extending-the-system)
   - [Example Usage](#example-usage)
5. [API Reference](#api-reference)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Future Enhancements](#future-enhancements)
9. [License](#license)

---

## System Architecture
The ForexPro Trading System is modular, consisting of three core components that form a cohesive pipeline:

1. **Synthetic Data Generation**:
   - Generates realistic 5-minute OHLCV (Open, High, Low, Close, Volume) data for EUR/USD and GBP/USD.
   - Simulates market dynamics, including volatility clustering, economic shocks, and correlated price movements.
   - Outputs data to `synthetic_forex_data.csv`.

2. **Feature Engineering**:
   - Enriches the synthetic data with technical indicators, microstructure features, and sentiment modeling.
   - Performs feature selection to remove multicollinearity and normalizes features for RL training.
   - Saves the enriched dataset to `featured_forex_data.csv` and generates validation plots.

3. **Reinforcement Learning Environment**:
   - Implements a custom Gym environment (`ForexProEnv`) for forex trading.
   - Trains a TQC-based RL agent using curriculum learning and hyperparameter optimization.
   - Evaluates the agent on financial metrics (e.g., Sharpe ratio, win rate) and renders trading plots.

The pipeline is designed to be extensible, allowing developers to customize data generation, add new features, or integrate alternative RL algorithms.

---

## Components

### Synthetic Data Generation
**File**: `generate_synthetic_data.py`

**Purpose**: Creates synthetic forex data to simulate realistic market conditions for training and testing trading strategies.

**Key Features**:
- Generates 5-minute OHLCV data from January 1, 2023, to January 1, 2025.
- Simulates volatility clustering, session-based volatility (higher from 8 AM to 5 PM), and economic shocks.
- Models correlation (0.8) between EUR/USD and GBP/USD.
- Assigns market regimes (`normal`, `volatile`, `crisis`) based on volatility and shocks.
- Computes technical indicators (e.g., RSI, MACD, Bollinger Bands) and custom features (e.g., fractal dimension).
- Saves data to CSV and plots price series.

**Dependencies**:
- `pandas`, `numpy`, `matplotlib`, `ta` (technical analysis library)

**Output**:
- `synthetic_forex_data.csv`: OHLCV data with regime labels.
- `synthetic_forex_plot.png`: Plot of EUR/USD and GBP/USD close prices.

**Known Issues**:
- Bug in `TSIIndicator` assignment (`df['tii'] = TSIIndicator` should be `df['tii'] = TSIIndicator(df['eurusd_close'], window_slow=25, window_fast=13).tsi()`).
- Hardcoded parameters limit flexibility.

### Feature Engineering
**File**: `feature_engineering.py`

**Purpose**: Enriches synthetic data with technical, microstructure, and sentiment features, preparing it for RL training.

**Key Features**:
- Loads `synthetic_forex_data.csv` and validates required columns.
- Adds:
  - Technical indicators (e.g., RSI, MACD, ATR, Bollinger Bands, ADX, Stochastic, TSI).
  - Short-window indicators for scalping (e.g., 5-period RSI).
  - Lagged features (e.g., RSI lags 1 and 2).
  - Volume features (e.g., volume z-scores, Accumulation/Distribution).
  - Fourier Transform (FFT) amplitudes for price cycles.
  - Microstructure features (e.g., bid-ask spread, slippage, order book imbalance).
  - News sentiment using an Ornstein-Uhlenbeck process.
  - Time-based features (sin/cos encodings for hour and day).
  - Risk management features (stop loss, take profit, position size).
  - Binary target for scalping (price increase > 2 pips).
- Performs feature selection to remove highly correlated features (threshold: 0.95).
- Normalizes features using `StandardScaler` and clips outliers.
- Generates validation plots for price, RSI, ATR, and bid-ask spread.

**Dependencies**:
- `pandas`, `numpy`, `matplotlib`, `ta`, `sklearn.preprocessing`

**Output**:
- `featured_forex_data.csv`: Enriched dataset with normalized features.
- Plots in `plots/` directory: `eurusd_regimes.png`, `rsi_comparison.png`, `atr_spread.png`.

**Known Issues**:
- Arbitrary clipping thresholds (e.g., ATR at 70th percentile, features at [-3, 3]).
- FFT window size (12 bars) is hardcoded.

### Reinforcement Learning Environment
**File**: `rl_trading.py`

**Purpose**: Implements a custom Gym environment for forex trading and trains a TQC-based RL agent to optimize trading decisions.

**Key Features**:
- **ForexProEnv**:
  - Observation space: Window of feature values (30 × feature_count).
  - Action space: 4D vector (position, size, stop-loss adjustment, take-profit adjustment).
  - Simulates trading with slippage, fees (0.00002), and commission (0.0001).
  - Tracks capital, P/L, and market regimes.
  - Complex reward function combining P/L, Sharpe ratio, drawdown penalties, and regime adjustments.
  - Renders price and P/L plots.
- **CurriculumWrapper**:
  - Adjusts difficulty (volatility, spread, max trades) based on performance.
  - Three difficulty levels with increasing complexity.
- **FinancialMetricsCallback**:
  - Tracks Sharpe ratio, Sortino ratio, drawdown, win rate, and mean reward.
- **Hyperparameter Optimization**:
  - Uses Optuna to tune TQC parameters (e.g., learning rate, batch size).
- **Training**:
  - Trains for 5 million timesteps with 2 parallel environments.
  - Saves checkpoints and the final model (`tqc_forex_advanced`).
- **Evaluation**:
  - Runs 10 episodes on validation data, computing P/L, win rate, Sharpe/Sortino ratios, and max drawdown.
  - Renders trading (`trading_plot.png`) and P/L (`pl_plot.png`) plots.

**Dependencies**:
- `gymnasium`, `stable_baselines3`, `sb3_contrib`, `optuna`, `psutil`, `matplotlib`, `numpy`, `pandas`

**Output**:
- `featured_forex_data.csv`: Input dataset.
- `logs/`: Training logs, monitor files, and checkpoints.
- `tqc_forex_advanced`: Trained TQC model.
- `trading_plot.png`, `pl_plot.png`: Evaluation plots.

**Known Issues**:
- Complex reward function may lead to unstable training.
- Limited evaluation (10 episodes) may not capture robustness.
- Hardcoded parameters (e.g., fees, capital).

---

## Installation and Setup
To set up the ForexPro Trading System, follow these steps:

1. **Prerequisites**:
   - Python 3.8 or higher.
   - pip package manager.
   - Git (optional, for cloning repositories).

3. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```plaintext
   pandas>=2.0.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   ta>=0.10.0
   scikit-learn>=1.2.0
   gymnasium>=0.29.0
   stable-baselines3>=2.0.0
   sb3-contrib>=2.0.0
   optuna>=3.1.0
   psutil>=5.9.0
   ```
   Install using:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**:
   ```python
   import pandas, numpy, matplotlib, ta, sklearn, gymnasium, stable_baselines3, sb3_contrib, optuna, psutil
   print("All dependencies installed successfully!")
   ```

6. **Directory Structure**:
   Ensure the following structure:
   ```
   forexpro-trading/
   ├── generate_synthetic_data.py
   ├── feature_engineering.py
   ├── rl_trading.py
   ├── logs/               # Created during training
   ├── plots/              # Created during feature engineering
   ├── requirements.txt
   ```

---

## Integration in Python
The ForexPro Trading System is designed for seamless integration into Python projects. Below are instructions for running the pipeline, extending the system, and using it in custom applications.

### Running the Pipeline
To execute the full pipeline (data generation, feature engineering, RL training), run the scripts sequentially:

1. **Generate Synthetic Data**:
   ```bash
   python generate_synthetic_data.py
   ```
   - Outputs `synthetic_forex_data.csv` and `synthetic_forex_plot.png`.

2. **Perform Feature Engineering**:
   ```bash
   python feature_engineering.py
   ```
   - Requires `synthetic_forex_data.csv`.
   - Outputs `featured_forex_data.csv` and plots in `plots/`.

3. **Train and Evaluate RL Model**:
   ```bash
   python rl_trading.py
   ```
   - Requires `featured_forex_data.csv`.
   - Outputs logs, checkpoints, model (`tqc_forex_advanced`), and evaluation plots.

### Extending the System
The system is modular, allowing developers to extend each component:

1. **Custom Data Generation**:
   - Modify `generate_synthetic_data.py` to adjust parameters (e.g., `base_volatility`, `correlation`).
   - Add new currency pairs by extending the data generation loop.
   - Example: Add USD/JPY with a new base price and correlation matrix.

2. **New Features**:
   - In `feature_engineering.py`, add indicators (e.g., Ichimoku Cloud via `ta`) or custom microstructure features.
   - Update `select_features` to include new features in the priority list.
   - Example: Add a volatility index feature:
     ```python
     df['vix'] = df['eurusd_close'].rolling(20).std() * np.sqrt(252)
     ```

3. **Alternative RL Algorithms**:
   - In `rl_trading.py`, replace TQC with PPO or SAC from `stable_baselines3`.
   - Example: Use PPO:
     ```python
     from stable_baselines3 import PPO
     model = PPO("MlpPolicy", env_train, learning_rate=3e-4, verbose=1)
     ```

4. **Real Data Integration**:
   - Replace synthetic data with real forex data (e.g., from MetaTrader, OANDA, or Dukascopy).
   - Update `feature_engineering.py` to handle real data formats (e.g., timestamp parsing).

### Example Usage
Below is an example of integrating the ForexPro system into a Python application to load a trained model and execute trades on new data.

```python
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC
from rl_trading import ForexProEnv, make_env

# Load new data (e.g., real or synthetic)
df = pd.read_csv('featured_forex_data.csv', parse_dates=['timestamp'])

# Create environment
env = DummyVecEnv([make_env(df, 0, seed=42)])

# Load trained model
model = TQC.load("tqc_forex_advanced")

# Run trading simulation
obs, _ = env.reset()
done = False
trades = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if 'pl' in info and info['pl'] != 0:
        trades.append(info['pl'])

# Print results
print(f"Total trades: {len(trades)}")
print(f"Mean P/L per trade: {sum(trades) / len(trades):.2f} USD" if trades else "No trades executed")

# Render plots
env.envs[0].render()
```

Save the above code as `forexpro_example.py` and run:
```bash
python forexpro_example.py
```

---

## API Reference
Below are key functions and classes for integrating the ForexPro system.

### Synthetic Data Generation
- **Function**: `create_features(df)`
  - **Parameters**:
    - `df`: `pandas.DataFrame` with OHLCV and regime columns.
  - **Returns**: `pandas.DataFrame` with technical indicators and custom features.
  - **Example**:
    ```python
    df = pd.read_csv('synthetic_forex_data.csv')
    df = create_features(df)
    ```

### Feature Engineering
- **Function**: `create_features(df, binary_target=True)`
  - **Parameters**:
    - `df`: `pandas.DataFrame` with OHLCV and regime columns.
    - `binary_target`: `bool`, whether to include a binary target (default: `True`).
  - **Returns**: `pandas.DataFrame` with enriched, normalized features.
  - **Example**:
    ```python
    df = pd.read_csv('synthetic_forex_data.csv', parse_dates=['timestamp'])
    df = create_features(df, binary_target=True)
    df.to_csv('featured_forex_data.csv')
    ```

- **Function**: `plot_features(df, n_bars=1000, save_path='plots')`
  - **Parameters**:
    - `df`: `pandas.DataFrame` with features.
    - `n_bars`: `int`, number of bars to plot (default: 1000).
    - `save_path`: `str`, directory for saving plots (default: `'plots'`).
  - **Returns**: None, saves plots to `save_path`.

### Reinforcement Learning Environment
- **Class**: `ForexProEnv`
  - **Initialization**:
    ```python
    env = ForexProEnv(df, window_size=30, max_episode_steps=5000, fees=0.00002)
    ```
    - `df`: `pandas.DataFrame` with features.
    - `window_size`: `int`, observation window size.
    - `max_episode_steps`: `int`, max steps per episode.
    - `fees`: `float`, trading fees.
  - **Methods**:
    - `reset(seed=None, options=None)`: Resets the environment.
    - `step(action)`: Executes an action, returns `(obs, reward, done, truncated, info)`.
    - `render(mode='human')`: Renders price and P/L plots.
    - `seed(seed=None)`: Sets random seed.

- **Function**: `make_env(df, rank, seed=42)`
  - **Parameters**:
    - `df`: `pandas.DataFrame` with features.
    - `rank`: `int`, environment rank for parallelization.
    - `seed`: `int`, random seed.
  - **Returns**: Callable to create a `ForexProEnv` instance.

- **Function**: `create_pro_model(env, params=None)`
  - **Parameters**:
    - `env`: Gym environment.
    - `params`: `dict`, TQC hyperparameters (optional).
  - **Returns**: `TQC` model instance.

---

## Best Practices
1. **Data Validation**:
   - Ensure input data contains required columns (`eurusd_close`, `gbpusd_close`, etc.).
   - Validate data ranges to prevent negative prices or unrealistic values.

2. **Modular Development**:
   - Encapsulate custom features or indicators in separate functions for reusability.
   - Use configuration files for parameters (e.g., volatility, fees).

3. **Performance Optimization**:
   - Vectorize computations in data generation and feature engineering.
   - Use parallel environments (`DummyVecEnv`) for RL training.

4. **Model Evaluation**:
   - Evaluate the RL model on diverse market conditions (e.g., trending, ranging).
   - Monitor financial metrics (Sharpe, Sortino) to assess performance.

5. **Version Control**:
   - Track changes to data, features, and models using Git.
   - Save model checkpoints and log training metrics.

6. **Resource Management**:
   - Monitor CPU/memory usage with `psutil` during training.
   - Use cloud resources for large-scale training.

---

## Troubleshooting
- **Error: Missing columns in DataFrame**:
  - Verify that `synthetic_forex_data.csv` contains all required columns.
  - Check data generation script for correct column names.

- **Error: TSIIndicator not callable**:
  - Fix the bug in `generate_synthetic_data.py`:
    ```python
    df['tii'] = TSIIndicator(df['eurusd_close'], window_slow=25, window_fast=13).tsi()
    ```

- **Training instability**:
  - Simplify the reward function in `ForexProEnv.step`.
  - Reduce learning rate or increase `learning_starts` in TQC.

- **FileNotFoundError**:
  - Ensure `synthetic_forex_data.csv` and `featured_forex_data.csv` are in the correct directory.
  - Check file paths in scripts.

- **High memory usage**:
  - Reduce `buffer_size` in TQC or use fewer parallel environments.
  - Process data in chunks for large datasets.

---

## Future Enhancements
1. **Real Data Integration**:
   - Connect to forex data providers (e.g., OANDA, MetaTrader) for real-time or historical data.
   - Adapt feature engineering to handle real data formats.

2. **Alternative RL Algorithms**:
   - Experiment with PPO, SAC, or DDPG for different trading scenarios.
   - Implement ensemble methods combining multiple RL models.

3. **Dynamic Feature Selection**:
   - Use ML-based feature importance (e.g., SHAP) to dynamically select features.
   - Incorporate online feature updates during training.

4. **Backtesting Framework**:
   - Add a backtesting module to validate the RL model on historical data.
   - Include transaction cost sensitivity analysis.

5. **Cloud Deployment**:
   - Deploy the system on AWS/GCP for scalable training and real-time trading.
   - Integrate with trading APIs for live execution.

---

## License
The ForexPro Trading System is licensed under the MIT License. See the `LICENSE` file for details.

```
License

Copyright (c) 2025, Kuntahfx 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---
**All kinds of donations for the good of this project are welcomed with gratitude, and if u have benefited of found it useful in any way a word or two as appretiation are enough, and if u are an experienced programmer in MQL5 we can also work together on a high yield project purely MQL5 state of the art. I Great u all fellow AlgoTraders in the name of the creator 🙏**
