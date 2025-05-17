import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.utils import dropna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Custom Accumulation/Distribution Index
def acc_dist_index(high, low, close, volume):
    """Calculate Accumulation/Distribution Index"""
    clv = ((close - low) - (high - close)) / (high - low + 1e-6)
    acc_dist = (clv * volume).cumsum()
    return np.clip(acc_dist, -1e6, 1e6)

def compute_fft_features(close_prices, window):
    """Compute FFT amplitude for a rolling window"""
    n = len(close_prices)
    amplitudes = np.zeros(n)
    for i in range(window, n):
        window_data = close_prices[i-window:i]
        fft = np.fft.fft(window_data)
        freq_idx = np.argmax(np.abs(fft[1:window//2])) + 1
        amplitudes[i] = np.abs(fft[freq_idx])
    return amplitudes

def check_feature_correlations(df, threshold=0.95, priority_features=None):
    """Identify highly correlated features"""
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'regime_volatile', 'regime_crisis', 'target_return', 'target_binary']]
    corr_matrix = df[feature_cols].corr().abs()
    high_corr = np.where(corr_matrix > threshold)
    pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
             for i, j in zip(*high_corr) if i != j and i < j]
    print(f"Highly correlated pairs: {pairs}")
    return sorted(pairs, key=lambda x: x[2], reverse=True)

def select_features(df, threshold=0.95, priority_features=None):
    """Drop one feature from each highly correlated pair, prioritizing key features"""
    if priority_features is None:
        priority_features = ['tii', 'gbpusd_tii', 'rsi', 'gbpusd_rsi', 'atr', 'gbpusd_atr', 
                            'macd', 'bb_lower', 'gbpusd_bb_lower', 'eurusd_close', 'gbpusd_close', 
                            'rsi_short', 'gbpusd_rsi_short', 'macd_short', 'stoch', 'gbpusd_stoch']
    protected_features = {'eurusd_close', 'gbpusd_close', 'tii', 'bb_lower', 'gbpusd_bb_lower', 
                         'rsi_short', 'gbpusd_rsi_short', 'stoch', 'gbpusd_stoch', 'stop_loss', 'take_profit'}
    corr_pairs = check_feature_correlations(df, threshold, priority_features)
    to_drop = set()
    for f1, f2, corr in corr_pairs:
        if f1 not in to_drop and f2 not in to_drop:
            if f1 in protected_features and f2 in protected_features:
                print(f"Correlation {corr:.3f}: Both {f1} and {f2} are protected, keeping both")
                continue
            elif f1 in protected_features:
                keep, drop = f1, f2
            elif f2 in protected_features:
                keep, drop = f2, f1
            elif f1 in priority_features and f2 not in priority_features:
                keep, drop = f1, f2
            elif f2 in priority_features and f1 not in priority_features:
                keep, drop = f2, f1
            elif f1 in priority_features and f2 in priority_features:
                keep = f1 if priority_features.index(f1) < priority_features.index(f2) else f2
                drop = f2 if keep == f1 else f1
            else:
                keep, drop = f1, f2
            print(f"Correlation {corr:.3f}: Keeping {keep}, dropping {drop}")
            to_drop.add(drop)
    print(f"Dropping highly correlated features: {to_drop}")
    print(f"Retained features: {set(df.columns) - to_drop}")
    return df.drop(columns=to_drop, errors='ignore')

def normalize_features(df):
    """Normalize numerical features for ML using StandardScaler"""
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'regime_volatile', 'regime_crisis', 'target_return', 'target_binary']]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df[feature_cols] = df[feature_cols].clip(-3, 3)  # Adjusted clipping range
    return df

def create_features(df, binary_target=True):
    """
    Add technical indicators and custom features to forex data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with key forex data columns (e.g., eurusd_close, gbpusd_close, etc.)
    binary_target : bool
        If True, include a binary target column.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame enriched with technical and microstructure features.
    """
    # Log synthetic data stats
    print("Synthetic data stats:")
    print(df[['eurusd_high', 'eurusd_low', 'eurusd_close']].describe())
    print("EURUSD high-low range counts:")
    print((df['eurusd_high'] - df['eurusd_low']).value_counts().head())

    # Validate input data
    required_cols = ['eurusd_close', 'gbpusd_close', 'eurusd_high', 'eurusd_low',
                     'gbpusd_high', 'gbpusd_low', 'eurusd_volume', 'gbpusd_volume', 'regime']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")

    # Returns
    df['eurusd_return'] = df['eurusd_close'].pct_change()
    df['gbpusd_return'] = df['gbpusd_close'].pct_change()

    # Short-Window Indicators for Scalping
    df['rsi_short'] = RSIIndicator(df['eurusd_close'], window=5).rsi()
    df['gbpusd_rsi_short'] = RSIIndicator(df['gbpusd_close'], window=5).rsi()
    macd_short = MACD(df['eurusd_close'], window_slow=8, window_fast=3, window_sign=5)
    df['macd_short'] = macd_short.macd()
    gbpusd_macd_short = MACD(df['gbpusd_close'], window_slow=8, window_fast=3, window_sign=5)
    df['gbpusd_macd_short'] = gbpusd_macd_short.macd()

    # Standard Indicators
    df['rsi'] = RSIIndicator(df['eurusd_close'], window=14).rsi()
    macd = MACD(df['eurusd_close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['atr'] = AverageTrueRange(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14).average_true_range()
    df['atr'] = df['atr'].clip(0, df['atr'].quantile(0.70)) + 1e-6
    df['atr'] = np.log1p(df['atr'])
    bb = BollingerBands(df['eurusd_close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['adx'] = ADXIndicator(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14).adx()
    df['stoch'] = StochasticOscillator(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14).stoch()
    df['tii'] = TSIIndicator(df['eurusd_close'], window_slow=25, window_fast=13).tsi()

    df['gbpusd_atr'] = AverageTrueRange(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14).average_true_range()
    df['gbpusd_atr'] = df['gbpusd_atr'].clip(0, df['gbpusd_atr'].quantile(0.70)) + 1e-6
    df['gbpusd_atr'] = np.log1p(df['gbpusd_atr'])
    df['gbpusd_rsi'] = RSIIndicator(df['gbpusd_close'], window=14).rsi()
    gbpusd_macd = MACD(df['gbpusd_close'], window_slow=26, window_fast=12, window_sign=9)
    df['gbpusd_macd'] = gbpusd_macd.macd()
    gbpusd_bb = BollingerBands(df['gbpusd_close'], window=20, window_dev=2)
    df['gbpusd_bb_upper'] = gbpusd_bb.bollinger_hband()
    df['gbpusd_bb_lower'] = gbpusd_bb.bollinger_lband()
    df['gbpusd_adx'] = ADXIndicator(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14).adx()
    df['gbpusd_stoch'] = StochasticOscillator(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14).stoch()
    df['gbpusd_tii'] = TSIIndicator(df['gbpusd_close'], window_slow=25, window_fast=13).tsi()

    # Lagged Features
    for lag in [1, 2]:
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        df[f'gbpusd_rsi_lag{lag}'] = df['gbpusd_rsi'].shift(lag)
        df[f'atr_lag{lag}'] = df['atr'].shift(lag)

    # Volume Features
    df['volume_zscore'] = (df['eurusd_volume'] - df['eurusd_volume'].rolling(50).mean()) / df['eurusd_volume'].rolling(50).std().replace(0, np.nan)
    df['gbpusd_volume_zscore'] = (df['gbpusd_volume'] - df['gbpusd_volume'].rolling(50).mean()) / df['gbpusd_volume'].rolling(50).std().replace(0, np.nan)
    df['acc_dist'] = acc_dist_index(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], df['eurusd_volume'])
    df['gbpusd_acc_dist'] = acc_dist_index(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], df['gbpusd_volume'])

    # Fourier Transform (5-minute timeframe: ~1 hour)
    window = 12
    df['fft_amplitude'] = compute_fft_features(df['eurusd_close'].values, window)
    df['gbpusd_fft_amplitude'] = compute_fft_features(df['gbpusd_close'].values, window)

    # Microstructure Features
    df['bid_ask_spread'] = 0.0001 + df['atr'] * 0.5 * (1 + 0.5 * (df['regime'] == 'crisis').astype(int))
    df['bid_ask_spread'] *= (1 + 0.2 * (df['eurusd_volume'] / df['eurusd_volume'].mean()))
    df['bid_ask_spread'] += np.random.normal(0, 0.00005, len(df))
    df['bid_ask_spread'] = df['bid_ask_spread'].clip(0.00005, 0.001)
    df['slippage'] = df['bid_ask_spread'] * np.random.uniform(0.5, 1.5, len(df))
    df['gbpusd_bid_ask_spread'] = 0.0001 + df['gbpusd_atr'] * 0.5 * (1 + 0.5 * (df['regime'] == 'crisis').astype(int))
    df['gbpusd_bid_ask_spread'] *= (1 + 0.2 * (df['gbpusd_volume'] / df['gbpusd_volume'].mean()))
    df['gbpusd_bid_ask_spread'] += np.random.normal(0, 0.00005, len(df))
    df['gbpusd_bid_ask_spread'] = df['gbpusd_bid_ask_spread'].clip(0.00005, 0.001)
    df['gbpusd_slippage'] = df['gbpusd_bid_ask_spread'] * np.random.uniform(0.5, 1.5, len(df))
    df['imbalance'] = (df['eurusd_volume'] - df['gbpusd_volume']) / (df['eurusd_volume'] + df['gbpusd_volume'] + 1e-6)

    # News Sentiment (Vectorized Ornstein-Uhlenbeck with shocks)
    theta, mu, sigma = 0.1, 0, 0.05
    n = len(df)
    dt = 1.0
    exp_term = np.exp(-theta * dt)
    sentiment = np.zeros(n)
    noise = np.random.normal(0, sigma * np.sqrt(1 - exp_term**2), n)
    sentiment[0] = mu
    for i in range(1, n):
        sentiment[i] = mu + exp_term * (sentiment[i-1] - mu) + noise[i]
        if df['regime'].iloc[i] == 'crisis' and np.random.random() < 0.1:
            sentiment[i] += np.random.uniform(-0.5, 0.5)
    df['news_sentiment'] = np.clip(sentiment, -1, 1)

    # Advanced Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        df['time_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['time_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

    # Session Feature
    df['session_overlap'] = ((df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 12)).astype(int)

    # Correlation
    df['correlation'] = df['eurusd_return'].rolling(100).corr(df['gbpusd_return'])

    # Regime Dummies
    df = pd.get_dummies(df, columns=['regime'], prefix='regime')
    for col in ['regime_normal', 'regime_volatile', 'regime_crisis']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(int)
    if 'regime_normal' in df.columns:
        df = df.drop(columns=['regime_normal'])

    # Handle NaNs
    short_window_cols = ['rsi', 'atr', 'bb_upper', 'bb_lower', 'volume_zscore', 'gbpusd_rsi', 
                        'gbpusd_atr', 'gbpusd_bb_upper', 'gbpusd_bb_lower', 'gbpusd_volume_zscore', 
                        'tii', 'gbpusd_tii', 'rsi_short', 'gbpusd_rsi_short', 'macd_short', 
                        'gbpusd_macd_short', 'rsi_lag1', 'rsi_lag2', 'gbpusd_rsi_lag1', 
                        'gbpusd_rsi_lag2', 'atr_lag1', 'atr_lag2']
    short_window_cols = [col for col in short_window_cols if col in df.columns]
    df[short_window_cols] = df[short_window_cols].ffill().bfill()
    
    long_window_cols = ['macd', 'stoch', 'adx', 'gbpusd_macd', 'gbpusd_stoch', 'gbpusd_adx', 'correlation']
    long_window_cols = [col for col in long_window_cols if col in df.columns]
    df[long_window_cols] = df[long_window_cols].fillna(0)
    
    df[['fft_amplitude', 'gbpusd_fft_amplitude']] = df[['fft_amplitude', 'gbpusd_fft_amplitude']].fillna(0)

    # Risk Management Features
    df['stop_loss'] = df['eurusd_close'] - 2 * df['atr']
    df['take_profit'] = df['eurusd_close'] + 4 * df['atr']
    df['position_size'] = 0.01 / (df['atr'] * 0.0001)

    # Scalping-Specific Target (2-pip threshold)
    pip_threshold = 0.0002
    df['target_return'] = df['eurusd_close'].pct_change().shift(-1)
    if binary_target:
        df['target_binary'] = ((df['eurusd_close'].shift(-1) - df['eurusd_close']) > pip_threshold).astype(int)

    # Feature selection
    df = select_features(df, threshold=0.95)

    # Normalize features
    df = normalize_features(df)

    # Drop rows with NaN targets
    df = df.dropna()

    # Validate features
    if 'rsi' in df.columns:
        print(f"RSI range: min={df['rsi'].min()}, max={df['rsi'].max()}")
    if 'gbpusd_rsi' in df.columns:
        print(f"GBPUSD RSI range: min={df['gbpusd_rsi'].min()}, max={df['gbpusd_rsi'].max()}")
    if 'atr' in df.columns:
        print(f"ATR range: min={df['atr'].min()}, max={df['atr'].max()}")
    if 'gbpusd_atr' in df.columns:
        print(f"GBPUSD ATR range: min={df['gbpusd_atr'].min()}, max={df['gbpusd_atr'].max()}")
    assert not df.isna().any().any(), "NaNs remain in the dataset"
    assert all(col in df.columns for col in ['regime_volatile', 'regime_crisis']), "Missing regime columns"

    return df

def plot_features(df, n_bars=1000, save_path='plots'):
    """Plot key features for validation"""
    os.makedirs(save_path, exist_ok=True)
    
    time_axis = df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp']

    # EURUSD Price with Regimes
    if 'eurusd_close' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_axis[-n_bars:], df['eurusd_close'][-n_bars:], label='EURUSD Close')
        for regime in ['volatile', 'crisis']:
            col = f'regime_{regime}'
            if col in df.columns:
                mask = df[col][-n_bars:] == 1
                ax.scatter(time_axis[-n_bars:][mask], df['eurusd_close'][-n_bars:][mask], 
                           label=f'{regime} regime', s=10)
        ax.set_title(f'EURUSD with Regimes (Last {n_bars} Bars)')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        plt.savefig(f'{save_path}/eurusd_regimes.png')
        plt.close()
    else:
        print("Warning: 'eurusd_close' not found, skipping EURUSD price plot")

    # RSI for EURUSD and GBPUSD
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'rsi' in df.columns:
        ax.plot(time_axis[-n_bars:], df['rsi'][-n_bars:], label='EURUSD RSI')
    if 'gbpusd_rsi' in df.columns:
        ax.plot(time_axis[-n_bars:], df['gbpusd_rsi'][-n_bars:], label='GBPUSD RSI', alpha=0.7)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-2, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'RSI for EURUSD and GBPUSD (Last {n_bars} Bars)')
    ax.set_ylabel('RSI (Normalized)')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'{save_path}/rsi_comparison.png')
    plt.close()

    # ATR and Bid-Ask Spread
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'atr' in df.columns:
        ax.plot(time_axis[-n_bars:], df['atr'][-n_bars:], label='EURUSD ATR')
    if 'gbpusd_atr' in df.columns:
        ax.plot(time_axis[-n_bars:], df['gbpusd_atr'][-n_bars:], label='GBPUSD ATR', alpha=0.7)
    if 'bid_ask_spread' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(time_axis[-n_bars:], df['bid_ask_spread'][-n_bars:], label='Bid-Ask Spread', color='r')
        ax2.set_ylabel('Bid-Ask Spread')
    ax.set_title(f'ATR and Bid-Ask Spread (Last {n_bars} Bars)')
    ax.set_ylabel('ATR (Normalized)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)
    plt.savefig(f'{save_path}/atr_spread.png')
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)

    try:
        df = pd.read_csv('synthetic_forex_data.csv', parse_dates=['timestamp'])
    except FileNotFoundError:
        raise FileNotFoundError("synthetic_forex_data.csv not found")

    df = create_features(df, binary_target=True)
    df.to_csv('featured_forex_data.csv', index=True)  # Ensure index is saved
    plot_features(df)