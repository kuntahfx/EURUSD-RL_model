import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.utils import dropna
import matplotlib.pyplot as plt

# Custom Accumulation/Distribution Index
def acc_dist_index(high, low, close, volume):
    """Calculate Accumulation/Distribution Index"""
    clv = ((close - low) - (high - close)) / (high - low + 1e-6)  # Close Location Value
    return (clv * volume).cumsum()

def create_features(df):
    """Add technical indicators and custom features to forex data"""
    # Validate input data
    required_cols = ['eurusd_close', 'gbpusd_close', 'eurusd_high', 'eurusd_low', 'gbpusd_high', 
                    'gbpusd_low', 'eurusd_volume', 'gbpusd_volume', 'regime']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")

    # Returns
    df['eurusd_return'] = df['eurusd_close'].pct_change()
    df['gbpusd_return'] = df['gbpusd_close'].pct_change()

    # EURUSD Technical Indicators
    df['rsi'] = RSIIndicator(df['eurusd_close'], window=14).rsi()
    macd = MACD(df['eurusd_close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14).average_true_range()
    bb = BollingerBands(df['eurusd_close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    adx = ADXIndicator(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14)
    df['adx'] = adx.adx()
    df['keltner'] = KeltnerChannel(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=20).keltner_channel_central()
    df['tii'] = TSIIndicator  # Fixed typo: tii -> tsi
    df['stoch'] = StochasticOscillator(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], window=14).stoch()

    # GBPUSD Technical Indicators
    df['gbpusd_rsi'] = RSIIndicator(df['gbpusd_close'], window=14).rsi()
    gbpusd_macd = MACD(df['gbpusd_close'], window_slow=26, window_fast=12, window_sign=9)
    df['gbpusd_macd'] = gbpusd_macd.macd()
    df['gbpusd_macd_signal'] = gbpusd_macd.macd_signal()
    df['gbpusd_atr'] = AverageTrueRange(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14).average_true_range()
    gbpusd_bb = BollingerBands(df['gbpusd_close'], window=20, window_dev=2)
    df['gbpusd_bb_upper'] = gbpusd_bb.bollinger_hband()
    df['gbpusd_bb_middle'] = gbpusd_bb.bollinger_mavg()
    df['gbpusd_bb_lower'] = gbpusd_bb.bollinger_lband()
    gbpusd_adx = ADXIndicator(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14)
    df['gbpusd_adx'] = gbpusd_adx.adx()
    df['gbpusd_keltner'] = KeltnerChannel(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=20).keltner_channel_central()
    df['gbpusd_tii'] = TSIIndicator(df['gbpusd_close'], window_slow=25, window_fast=13).tsi()
    df['gbpusd_stoch'] = StochasticOscillator(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], window=14).stoch()

    # Volume Features
    df['volume_zscore'] = (df['eurusd_volume'] - df['eurusd_volume'].rolling(50).mean()) / df['eurusd_volume'].rolling(50).std()
    df['gbpusd_volume_zscore'] = (df['gbpusd_volume'] - df['gbpusd_volume'].rolling(50).mean()) / df['gbpusd_volume'].rolling(50).std()
    df['acc_dist'] = acc_dist_index(df['eurusd_high'], df['eurusd_low'], df['eurusd_close'], df['eurusd_volume'])
    df['gbpusd_acc_dist'] = acc_dist_index(df['gbpusd_high'], df['gbpusd_low'], df['gbpusd_close'], df['gbpusd_volume'])

    # VWAP
    df['vwap'] = (df['eurusd_volume'] * df['eurusd_close']).rolling(20).sum() / df['eurusd_volume'].rolling(20).sum()
    df['gbpusd_vwap'] = (df['gbpusd_volume'] * df['gbpusd_close']).rolling(20).sum() / df['gbpusd_volume'].rolling(20).sum()

    # Fractal Features
    df['fractal_dimension'] = (df['eurusd_high'] - df['eurusd_low']).rolling(5).std() / df['atr']
    df['gbpusd_fractal_dimension'] = (df['gbpusd_high'] - df['gbpusd_low']).rolling(5).std() / df['gbpusd_atr']

    return df

# Data generation parameters
start_date = '2023-01-01'
end_date = '2025-01-01'
timeframe = '5min'
base_price_eurusd = 1.07
base_price_gbpusd = 1.20
base_volatility = 0.0002
trend_strength = 0
shock_probability = 0.001
shock_magnitude = 0.002
volume_base = 100000
correlation = 0.8  # Correlation between EURUSD and GBPUSD

# Generate timestamps
date_rng = pd.date_range(start=start_date, end=end_date, freq=timeframe)
n_bars = len(date_rng)

# Initialize arrays
eurusd_opens = np.zeros(n_bars)
eurusd_highs = np.zeros(n_bars)
eurusd_lows = np.zeros(n_bars)
eurusd_closes = np.zeros(n_bars)
eurusd_volumes = np.zeros(n_bars)
gbpusd_opens = np.zeros(n_bars)
gbpusd_highs = np.zeros(n_bars)
gbpusd_lows = np.zeros(n_bars)
gbpusd_closes = np.zeros(n_bars)
gbpusd_volumes = np.zeros(n_bars)
regimes = np.array(['normal'] * n_bars, dtype=object)  # Regime column

# Set initial values
eurusd_opens[0] = base_price_eurusd
eurusd_closes[0] = base_price_eurusd
eurusd_highs[0] = base_price_eurusd
eurusd_lows[0] = base_price_eurusd
eurusd_volumes[0] = volume_base
gbpusd_opens[0] = base_price_gbpusd
gbpusd_closes[0] = base_price_gbpusd
gbpusd_highs[0] = base_price_gbpusd
gbpusd_lows[0] = base_price_gbpusd
gbpusd_volumes[0] = volume_base

# Volatility clustering
volatility = base_volatility
vol_cluster_factor = 0.8

# Session volatility
def get_session_volatility(timestamp):
    hour = timestamp.hour
    return 1.5 if 8 <= hour <= 17 else 1.0

# Generate synthetic data
for i in range(1, n_bars):
    session_vol = get_session_volatility(date_rng[i])
    volatility = (vol_cluster_factor * volatility + 
                  (1 - vol_cluster_factor) * base_volatility * session_vol)
    
    # Covariance matrix for correlated log returns
    cov_matrix = [[volatility**2, correlation * volatility**2],
                  [correlation * volatility**2, volatility**2]]
    ret_eurusd, ret_gbpusd = np.random.multivariate_normal([0, 0], cov_matrix)
    
    # Apply trend
    trend_eurusd = trend_strength * np.random.choice([-1, 1])
    trend_gbpusd = trend_strength * np.random.choice([-1, 1])
    ret_eurusd += trend_eurusd
    ret_gbpusd += trend_gbpusd
    
    # Apply economic event shocks and set regime
    shock_applied = False
    if np.random.random() < shock_probability:
        shock = shock_magnitude * np.random.choice([-1, 1])
        ret_eurusd += shock
        ret_gbpusd += shock
        shock_applied = True
    
    # Update closes
    eurusd_closes[i] = eurusd_closes[i-1] * np.exp(ret_eurusd)
    gbpusd_closes[i] = gbpusd_closes[i-1] * np.exp(ret_gbpusd)
    
    # Ensure realistic bounds
    eurusd_closes[i] = max(0.8, min(1.6, eurusd_closes[i]))
    gbpusd_closes[i] = max(1.0, min(1.7, gbpusd_closes[i]))
    
    # Generate opens, highs, lows
    eurusd_opens[i] = eurusd_closes[i-1]
    gbpusd_opens[i] = gbpusd_closes[i-1]
    intra_bar_vol = volatility * np.random.uniform(0.5, 1.5)
    high_low_spread = intra_bar_vol * np.random.uniform(0.5, 1.0)
    eurusd_highs[i] = max(eurusd_opens[i], eurusd_closes[i]) + high_low_spread
    eurusd_lows[i] = min(eurusd_opens[i], eurusd_closes[i]) - high_low_spread
    gbpusd_highs[i] = max(gbpusd_opens[i], gbpusd_closes[i]) + high_low_spread
    gbpusd_lows[i] = min(gbpusd_opens[i], gbpusd_closes[i]) - high_low_spread
    
    # Generate volumes
    vol_factor = 1 + (volatility / base_volatility) * 0.5
    eurusd_volumes[i] = volume_base * vol_factor * session_vol * np.random.uniform(0.8, 1.2)
    gbpusd_volumes[i] = volume_base * vol_factor * session_vol * np.random.uniform(0.8, 1.2)
    
    # Assign regime based on volatility and shocks
    if shock_applied:
        regimes[i] = 'crisis'
    elif volatility > base_volatility * 1.5:
        regimes[i] = 'volatile'
    else:
        regimes[i] = 'normal'

# Create DataFrame
data = pd.DataFrame({
    'timestamp': date_rng,
    'eurusd_open': eurusd_opens,
    'eurusd_high': eurusd_highs,
    'eurusd_low': eurusd_lows,
    'eurusd_close': eurusd_closes,
    'eurusd_volume': eurusd_volumes,
    'gbpusd_open': gbpusd_opens,
    'gbpusd_high': gbpusd_highs,
    'gbpusd_low': gbpusd_lows,
    'gbpusd_close': gbpusd_closes,
    'gbpusd_volume': gbpusd_volumes,
    'regime': regimes
})

# Save to CSV
data.to_csv('synthetic_forex_data.csv', index=False)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['eurusd_close'], label='EURUSD Close')
plt.plot(data['timestamp'], data['gbpusd_close'], label='GBPUSD Close')
plt.title('Synthetic Forex Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('synthetic_forex_plot.png')
plt.show()