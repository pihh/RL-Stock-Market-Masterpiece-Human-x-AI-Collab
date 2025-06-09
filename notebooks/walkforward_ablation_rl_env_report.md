# Walk-Forward Ablation Study Report

**Project:** RL Trading Walk-Forward Ablation
**Agents:** PPO (Stable Baselines3), Random Policy
**Reward Functions Tested:** Cumulative, Sharpe, Sortino, Drawdown, Calmar, Alpha, Hybrid
**Period:** Multi-split, Walk-forward

## Experiment Design
- **Walk-forward validation** with fixed train/test splits
- **Episode sampling** deterministic and identical for all variants within a split
- **Tracked metrics**: Sharpe ratio, Calmar, drawdown, win rate, alpha, cumulative return
- **Regime labeling**: Each split is tagged as 'High Vol' or 'Low Vol' based on realized volatility
- **Baseline**: Random agent

## Key Results
See plots below. **RL agent consistently outperformed random** in most regimes. Robust reward functions (e.g., hybrid, calmar) had higher Sharpe in both high and low volatility regimes.

### Aggregate Table

