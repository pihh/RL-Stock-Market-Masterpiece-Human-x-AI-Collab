# RL Stock Market Masterpiece: Human x AI Collab

Welcome to our collaborative project—where **human intuition meets AI horsepower**—to build a live-trading Reinforcement Learning agent that can take on the stock market.

## 🚀 Project Mission

Design, build, and rigorously test a robust RL agent that can **beat the market** in live conditions. Every decision, from data pipeline to model to execution, will be grounded in transparency, statistical rigor, and reproducibility.

- **Tools:** Python, Alpaca API (paper trading for now), best-in-class open source libraries
- **Approach:** Scientific, iterative, no shortcuts—every choice documented and justified

## 🤝 Who We Are

**Filipe Sá:**  
Trader, engineer, and curious mind—doing the heavy lifting, coding, and live experiments.

**ChatGPT (OpenAI):**  
AI cofounder, brainstorming partner, and code-writing assistant—bringing research insights, code templates, and endless patience.

> This project is a genuine collab: every contribution is open, documented, and discussed. We believe that giving credit is just as important as getting results.

## 📚 Project Structure

1. **Data Pipeline:** Historical + live data acquisition, robust storage, feature engineering
2. **Universe Selection:** Filtering stocks for agent’s best shot at generalization
3. **RL Agent:** Training, backtesting, walk-forward validation
4. **Execution Engine:** Live trading infra, order management, risk controls
5. **Analysis & Research:** All results, metrics, and design decisions published and open for review

## 📖 Why Document Everything?

Science matters. We’ll comment on every major choice (feature, reward, model tweak, etc.), so **future us** (and anyone reading) will know not just _what_ we did, but _why_.

---

**Join us (virtually) on this journey.**  
Watch as a human and an AI build, break, and (hopefully) beat the market—together.

---

> _If you’re reading this and want to contribute or just follow along, check out the Issues, Discussions, or open a PR!_

## Key Findings (as of June 2025)

- Predictability via plain R² has low generalization across months (R²_train vs R²_test < 0.2)
- Using residual autocorrelation as a signal improves meta-model recall by +15%
- Stocks labeled as "easy" by tiny RL agents also yield higher Sharpe when used in full-scale RL training
- Contrastive predictability ranking outperforms direct R² regression when applied to month pairs