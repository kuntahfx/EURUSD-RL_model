import os
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import psutil
import random
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sb3_contrib import TQC
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

#####################################
# ForexPro Trading Environment
#####################################
class ForexProEnv(gym.Env):
    def __init__(self, df, window_size=30, max_episode_steps=5000, fees=0.00002, **kwargs):
        super().__init__()
        print(f"Available columns: {df.columns.tolist()}")
        required_cols = ['eurusd_close', 'atr', 'bid_ask_spread', 'regime_volatile',
                         'regime_crisis', 'position_size', 'stop_loss', 'take_profit']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")
        
        self.df = df.copy().reset_index(drop=True)
        for col in ['regime_volatile', 'regime_crisis']:
            self.df[col] = self.df[col].astype(float)
            
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.fees = fees
        self.commission = kwargs.get('commission', 0.0001)
        self.current_step = window_size
        self.episode_step = 0
        self.entry_step = 0
        self.position = 0.0
        self.lot_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.capital = 10000
        self.max_drawdown = 0.1
        self.daily_pl = 0.0
        self.daily_pl_reset_step = 0
        self.returns = deque(maxlen=288)
        
        # Define feature columns excluding 'Unnamed: 0', 'target_return', 'target_binary'
        self.feature_cols = [col for col in self.df.columns if col not in ['Unnamed: 0', 'target_return', 'target_binary']]
        
        # Set observation space with correct shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, len(self.feature_cols)),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))  # [position, size, sl_adj, tp_adj]
        self.atr = self.df['atr'].values
        self.reward_buffer = deque(maxlen=1000)
        self.render_data = {'prices': [], 'positions': [], 'pl': [], 'regimes': []}
        
        for _ in range(1000):
            self.reward_buffer.append(0.0)
        
        self.params = {}
        self.np_random = None  # Local random number generator, initialized in seed
    
    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        max_start = max(self.window_size, len(self.df) - self.max_episode_steps - 1)
        self.current_step = self.np_random.randint(self.window_size, max_start)
        self.episode_step = 0
        self.entry_step = 0
        self.position = 0.0
        self.lot_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.capital = 10000
        self.daily_pl = 0.0
        self.daily_pl_reset_step = self.current_step
        self.returns = deque(maxlen=288)
        self.reward_buffer = deque(maxlen=1000)
        for _ in range(1000):
            self.reward_buffer.append(0.0)
        self.render_data = {'prices': [], 'positions': [], 'pl': [], 'regimes': []}
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current observation using only feature columns."""
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step][self.feature_cols].values
        return obs
    
    def _calculate_position_size(self, size_action):
        base_size = self.df.loc[self.current_step, 'position_size']
        adjusted_size = (size_action + 1) * base_size * 0.5
        self.lot_size = np.clip(adjusted_size * (self.capital / 10000), 0.1, 10)
        return self.lot_size
    
    def _calculate_stop_loss(self, sl_action):
        base_sl = self.df.loc[self.current_step, 'stop_loss']
        return base_sl * np.clip((sl_action + 1) * 0.5 + 0.5, 0.5, 2)
    
    def _calculate_take_profit(self, tp_action):
        base_tp = self.df.loc[self.current_step, 'take_profit']
        return base_tp * np.clip((tp_action + 1) * 0.5 + 0.5, 1, 4)
    
    def _calculate_risk(self):
        var = self.capital * 0.01
        drawdown_adj = 1 - min(abs(self.daily_pl) / (self.capital * 0.03), 1)
        volatility_adj = min(self.atr[self.current_step] / (np.mean(self.atr[:self.current_step]) + 1e-6), 2)
        regime_adj = 0.5 if self._market_state() == 'crisis' else 1.0
        return var * drawdown_adj * volatility_adj * regime_adj
    
    def _calculate_sharpe(self):
        if len(self.returns) < 10:
            return 1.0
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-6
        return mean_return / std_return * np.sqrt(288)
    
    def _market_state(self):
        if self.df.loc[self.current_step, 'regime_crisis'] > 0.5:
            return 'crisis'
        elif self.df.loc[self.current_step, 'regime_volatile'] > 0.5:
            return 'volatile'
        return 'normal'
    
    def _slippage_model(self):
        """Calculate slippage with local random number generator."""
        base_slippage = self.np_random.exponential(0.00005)
        regime_factor = 2.0 if self._market_state() == 'crisis' else 1.0
        return np.clip(base_slippage * regime_factor, 0, 0.0005)
    
    def step(self, action):
        price = self.df.loc[self.current_step, 'eurusd_close']
        spread = self.df.loc[self.current_step, 'bid_ask_spread']
        atr = self.atr[self.current_step]
        reward = 0.0
        done = False
        
        if (self.current_step - self.daily_pl_reset_step) >= 288:
            self.daily_pl = 0.0
            self.daily_pl_reset_step = self.current_step
        
        position_action = np.clip(action[0], -1, 1)
        lot_size = self._calculate_position_size(action[1])
        stop_loss = self._calculate_stop_loss(action[2])
        take_profit = self._calculate_take_profit(action[3])
        
        risk_budget = self._calculate_risk() / (self.capital + 1e-6)
        position_action = np.clip(position_action, -risk_budget, risk_budget)
        
        auto_close = False
        if self.position != 0:
            if self.position > 0:
                if price <= self.entry_price - stop_loss:
                    auto_close = True
                    reward -= 0.5 * 1e-4
                elif price >= self.entry_price + take_profit:
                    auto_close = True
                    reward += 1.0 * 1e-4
            elif self.position < 0:
                if price >= self.entry_price + stop_loss:
                    auto_close = True
                    reward -= 0.5 * 1e-4
                elif price <= self.entry_price - take_profit:
                    auto_close = True
                    reward += 1.0 * 1e-4
        
        if auto_close or (self.episode_step - self.entry_step >= 5 and abs(position_action) < 0.1 and self.position != 0):
            if self.position != 0:
                pl = ((price - self.entry_price) * self.position - spread - self.commission -
                      self.fees - self._slippage_model()) * self.lot_size
                self.daily_pl += pl
                self.capital += pl
                self.returns.append(pl / (self.capital + 1e-6))
                raw_reward = pl / (atr + 1e-6)
                sharpe = self._calculate_sharpe()
                reward += np.clip(raw_reward * sharpe * 1e-4, -500 * 1e-4, 500 * 1e-4) + (1e-4 if pl > 0 else -0.5e-4)
                self.position = 0
                self.lot_size = 0
                self.stop_loss = 0
                self.take_profit = 0
        elif abs(position_action) >= 0.1 and self.position == 0:
            direction = np.sign(position_action)
            slippage = self._slippage_model() * (-direction)
            executed_price = price + slippage
            self.position = position_action
            self.lot_size = lot_size
            self.entry_price = executed_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.entry_step = self.episode_step
            reward -= 0.1 * 1e-4  # Entry cost
        
        if self.position != 0:
            unrealized_pl = (price - self.entry_price) * self.position * self.lot_size
            reward += 0.001 * 1e-4 * unrealized_pl / (atr + 1e-6)
            drawdown = max(0, self.entry_price - price) if self.position > 0 else max(0, price - self.entry_price)
            reward -= 0.003 * 1e-4 * drawdown / (atr + 1e-6)
        
        state = self._market_state()
        regime_adj = -0.1 * self.atr[self.current_step] if state == 'crisis' else 0.05 * self.atr[self.current_step] if state == 'volatile' else 0
        reward += abs(self.position) * regime_adj * 1e-4
        reward -= 0.01 * 1e-4 * self.episode_step / 100
        
        self.reward_buffer.append(reward)
        reward_mean = np.mean(self.reward_buffer)
        reward_std = np.std(self.reward_buffer) + 1e-6
        normalized_reward = (reward - reward_mean) / reward_std
        
        drawdown_ratio = abs(self.daily_pl) / (self.capital * self.max_drawdown)
        if drawdown_ratio > 1:
            done = True
            normalized_reward -= 1.0
        else:
            normalized_reward -= 0.5 * drawdown_ratio
        
        self.current_step += 1
        self.episode_step += 1
        done = done or self.current_step >= len(self.df) - 1 or self.episode_step >= self.max_episode_steps
        
        self.render_data['prices'].append(price)
        self.render_data['positions'].append(self.position)
        self.render_data['pl'].append(self.capital - 10000)
        self.render_data['regimes'].append(state)
        
        obs = self._get_observation()
        return obs, normalized_reward, done, False, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(12, 6))
            prices = self.render_data['prices']
            positions = self.render_data['positions']
            regimes = self.render_data['regimes']
            plt.plot(prices, label='EURUSD Close')
            buy_points = [i for i, p in enumerate(positions) if p > 0]
            sell_points = [i for i, p in enumerate(positions) if p < 0]
            plt.scatter(buy_points, [prices[i] for i in buy_points],
                        c='green', marker='^', label='Long')
            plt.scatter(sell_points, [prices[i] for i in sell_points],
                        c='red', marker='v', label='Short')
            for i, regime in enumerate(regimes):
                if regime == 'crisis':
                    plt.axvspan(i-0.5, i+0.5, color='red', alpha=0.1)
                elif regime == 'volatile':
                    plt.axvspan(i-0.5, i+0.5, color='blue', alpha=0.1)
            plt.title('Forex Trading Environment')
            plt.xlabel('Step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig('trading_plot.png')
            plt.close()
            
            plt.figure(figsize=(12, 4))
            plt.plot(self.render_data['pl'], label='P/L')
            plt.title('Profit/Loss')
            plt.xlabel('Step')
            plt.ylabel('P/L (USD)')
            plt.legend()
            plt.grid(True)
            plt.savefig('pl_plot.png')
            plt.close()

#####################################
# CurriculumWrapper for adaptive difficulty
#####################################
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_difficulty = 0
        self.difficulty_levels = [
            {'vol_scale': 0.5, 'max_trades': 10, 'spread_scale': 0.5},
            {'vol_scale': 1.0, 'max_trades': 20, 'spread_scale': 1.0},
            {'vol_scale': 2.0, 'max_trades': 30, 'spread_scale': 1.5}
        ]
        self.episode_rewards = []
    
    @property
    def num_envs(self):
        return 1
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def reset(self, **kwargs):
        if len(self.episode_rewards) > 0 and np.mean(self.episode_rewards[-5:]) < -0.5:
            self.current_difficulty = max(0, self.current_difficulty - 1)
        self.env.set_params(**self.difficulty_levels[self.current_difficulty])
        print(f"Difficulty set to {self.current_difficulty}: {self.difficulty_levels[self.current_difficulty]}")
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            self.episode_rewards.append(info.get('episode', {}).get('r', 0))
            if len(self.episode_rewards) >= 5 and np.mean(self.episode_rewards[-5:]) > 0.5:
                self.current_difficulty = min(2, self.current_difficulty + 1)
                self.env.set_params(**self.difficulty_levels[self.current_difficulty])
                print(f"Difficulty increased to {self.current_difficulty}: {self.difficulty_levels[self.current_difficulty]}")
        return obs, reward, done, truncated, info
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)

#####################################
# FinancialMetricsCallback for tracking metrics
#####################################
class FinancialMetricsCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.sharpe_ratios = []
        self.drawdowns = []
        self.sortino_ratios = []
        self.win_rates = []
        self.trades = []
        self.wins = []
        self.rewards = []

    def _on_step(self):
        super()._on_step()
        env = self.eval_env.envs[0]
        returns = np.array(env.returns)
        if len(returns) > 10:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(288)
            downside_returns = returns[returns < 0]
            sortino = (np.mean(returns) / (np.std(downside_returns) + 1e-6) * np.sqrt(288)
                       if len(downside_returns) > 0 else sharpe)
            self.sharpe_ratios.append(sharpe)
            self.sortino_ratios.append(sortino)
        drawdown = abs(env.daily_pl) / env.capital
        self.drawdowns.append(drawdown)
        if env.position == 0 and env.lot_size > 0:
            pl = env.daily_pl - sum(self.trades[-1:]) if self.trades else env.daily_pl
            self.trades.append(pl)
            if pl > 0:
                self.wins.append(1)
            win_rate = len(self.wins) / len(self.trades) if self.trades else 0
            self.win_rates.append(win_rate)
            self.rewards.append(np.mean(env.reward_buffer))
            print(f"Eval Win Rate: {win_rate:.2%}, Trades: {len(self.trades)}, Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, Mean Reward: {np.mean(env.reward_buffer):.4f}")
        return True

#####################################
# Helper functions
#####################################
def make_env(df, rank, seed=42):
    def _init():
        try:
            env = ForexProEnv(df, window_size=30, max_episode_steps=5000)
            env = CurriculumWrapper(env)
            env.seed(seed + rank)  # Now works with added seed method
            return env
        except Exception as e:
            print(f"Error in subprocess {rank}: {e}")
            raise
    return _init

def optimize_hyperparams(env_train, env_val):
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        tau = trial.suggest_float('tau', 0.005, 0.05)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        ent_coef = trial.suggest_float('ent_coef', 1e-5, 1e-2, log=True)
        train_freq = trial.suggest_int('train_freq', 500, 2000)
        noise_sigma = trial.suggest_float('noise_sigma', 0.05, 0.2)
        policy_kwargs = {
            'net_arch': {'pi': [128, 128], 'qf': [256, 128]},
            'n_critics': 2
        }
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env_train.action_space.shape),
            sigma=noise_sigma * np.ones(env_train.action_space.shape)
        )
        model = TQC(
            "MlpPolicy",
            env_train,
            learning_rate=lr,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=batch_size,
            train_freq=(train_freq, 'step'),
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=42
        )
        model.learn(total_timesteps=200000, callback=eval_callback)
        mean_reward = np.mean([r for r in eval_callback.evaluations_results[-5:] if r is not None])
        return mean_reward

    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(objective, n_trials=50, n_jobs=1)
    return study.best_params

def create_pro_model(env, params=None):
    if params is None:
        params = {'lr': 3e-5, 'batch_size': 1024, 'tau': 0.02, 'gamma': 0.999, 
                  'ent_coef': 'auto', 'train_freq': 1000, 'noise_sigma': 0.1}
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=params['noise_sigma'] * np.ones(env.action_space.shape)
    )
    policy_kwargs = {
        'net_arch': {'pi': [128, 128], 'qf': [256, 128]},
        'n_critics': 2
    }
    model = TQC(
        "MlpPolicy",
        env,
        learning_rate=params['lr'],
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=params['batch_size'],
        train_freq=(params['train_freq'], 'step'),
        tau=params['tau'],
        gamma=params['gamma'],
        ent_coef=params['ent_coef'],
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42
    )
    return model

#####################################
# Main execution
#####################################
if __name__ == "__main__":
    try:
        df = pd.read_csv('featured_forex_data.csv', parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        raise FileNotFoundError("featured_forex_data.csv not found")
    df = df.reset_index(drop=True)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:]

    print(f"CPU cores: {os.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    os.makedirs('./logs/', exist_ok=True)
    if not os.access('./logs/', os.W_OK):
        raise PermissionError("Cannot write to ./logs/ directory")

    env_train = DummyVecEnv([make_env(df_train, i) for i in range(2)])
    env_train = VecMonitor(env_train, filename='./logs/monitor_train.csv')
    env_val = DummyVecEnv([make_env(df_val, 0)])
    env_val = VecMonitor(env_val, filename='./logs/monitor_val.csv')

    eval_callback = FinancialMetricsCallback(
        env_val,
        best_model_save_path='./logs/best_model',
        log_path='./logs/',
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // 2,
        save_path='./logs/',
        name_prefix='tqc_forex'
    )

    best_params = optimize_hyperparams(env_train, env_val)
    print(f"Best hyperparameters: {best_params}")
    
    env_train = DummyVecEnv([make_env(df_train, i) for i in range(2)])
    env_train = VecMonitor(env_train, filename='./logs/monitor_train_final.csv')
    model = create_pro_model(env_train, best_params)
    
    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save("tqc_forex_advanced")
    
    eval_env = ForexProEnv(df_val, window_size=30, max_episode_steps=len(df_val))
    episode_returns = []
    episode_trades = []
    episode_wins = []
    for episode in range(10):
        obs, _ = eval_env.reset()
        returns = []
        trades = 0
        wins = 0
        max_steps = min(5000, len(df_val) - eval_env.current_step)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            if eval_env.position == 0 and eval_env.lot_size > 0:
                pl = info.get('pl', 0) - sum(returns[-1:]) if returns else info.get('pl', 0)
                returns.append(pl)
                trades += 1
                if pl > 0:
                    wins += 1
            if done or truncated:
                break
        episode_returns.append(returns)
        episode_trades.append(trades)
        episode_wins.append(wins)
        eval_env.render()
    
    all_returns = [r for eps in episode_returns for r in eps]
    total_trades = sum(episode_trades)
    total_wins = sum(episode_wins)
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-6) * np.sqrt(288)
    downside_returns = [r for r in all_returns if r < 0]
    sortino = (np.mean(all_returns) / (np.std(downside_returns) + 1e-6) * np.sqrt(288)
               if len(downside_returns) > 0 else sharpe)
    max_drawdown = max([abs(r) / 10000 for r in all_returns]) if all_returns else 0
    print(f"Evaluation over {len(episode_returns)} episodes:")
    print(f"Mean P/L per trade: {np.mean(all_returns):.2f} USD")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Trades: {total_trades}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")