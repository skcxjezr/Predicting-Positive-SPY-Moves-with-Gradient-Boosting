# Predicting Positive SPY Moves with Gradient Boosting

_A compact, reproducible study that builds and evaluates a next-day **up/down** classifier for **SPY** using gradient boosting and common technical features._


Author Info:

- Kecheng Shi, ks4327@columbia.edu, (212) 814-1023
- Please contact me with any questions, concerns, or recommendations

## 🧭 Overview

Short-horizon equity returns are noisy, but engineered signals + robust validation can still uncover small, usable edges.  
This notebook:

- Downloads ~5 years of **SPY** daily OHLCV via **yfinance**
- Engineers a focused set of technical features (volatility, skew, MACD, Bollinger %B, lags, etc.)
- Trains a **gradient boosting** model (XGBoost) to predict **tomorrow’s positive return** (label ∈ {0,1})
- Tunes hyperparameters with **time-series splits**
- Calibrates a decision **threshold** (Youden’s J; ~0.48 in the notebook run)
- Runs a simple **long-only backtest** from the probability stream and compares to buy-and-hold


---

## ✨ Key Results (from the included run)

- **Hold-out ROC AUC:** ~**0.68**  
- **Backtest (long-only, no costs):**  
  - Annualized **Sharpe:** ~**0.8** (252-day convention)  
  - **Max drawdown:** strategy ~**−6–7%** vs buy-and-hold ~**−18–19%**  
- **Most useful features** in the final, compact model include: `Vol_5`, `Skew_5`, `MACD`, `BB_pctb`, short-horizon `Return` lags, `RSI_14`.

Your exact numbers may vary run-to-run and with data freshness.

---

## 🛠️ Setup

### 1) Python environment

```bash
# fresh venv/conda recommended
pip install -U pandas numpy matplotlib scikit-learn xgboost yfinance
# optional, if you want Jupyter locally:
pip install -U jupyterlab
```

### 2) Open the notebook

```bash
jupyter lab
# then open: Predicting Positive SPY Moves with Gradient Boosting.ipynb
```

---

## ▶️ How to run

1. **Run all cells** top-to-bottom.  
2. The notebook will:
   - Pull SPY daily bars (5y window by default)
   - Build features (rolling stats, MACD, Bollinger %B, RSI, ROC, return lags, etc.)
   - Create the **label**: `Label = 1` if `Close[t+1] > Close[t]`, else `0`
   - Split chronologically (80/20), tune an **XGBClassifier** with `TimeSeriesSplit`
   - Evaluate on the **final hold-out**
   - Convert probabilities to signals using the tuned **threshold**
   - **Backtest** a 1/0 long-only strategy and print summary stats/plot

---

## 🔍 Methodology (concise)

**Data**  
- Source: Yahoo Finance via `yfinance`  
- Asset: `SPY` (can be changed)  
- Frequency: Daily (U.S. business days using a holiday calendar)  
- Lookback: ~5 years (default; adjustable)

**Features (examples)**  
- Rolling returns & dispersion: `Return`, `Vol_5/10/20`, `Skew_5/10/20`, `Kurt_5/10/20`  
- **Momentum / oscillators:** `RSI_14`, `ROC_10`, `MACD`, `MACD_Signal`  
- **Bands:** Bollinger mid/±2σ and **`BB_pctb`**  
- **Lags:** `Return_lag_1..4`  
- (Optional) simple volume trend: `Vol_MA_20`

**Label**  
- Binary: **1** if next-day close > today’s close; else **0**.

**Modeling**  
- Estimator: **XGBoost (binary:logistic)**  
- Search: `GridSearchCV` with **`TimeSeriesSplit`**  
- Metrics: **ROC AUC**, confusion matrix, classification report  
- Threshold: tuned from ROC statistics (e.g., **Youden’s J**)

**Backtest (toy)**  
- Signal: go **long (1)** when `P(up) ≥ threshold`, else **flat (0)`  
- Returns: next-day close-to-close * position (no leverage/costs/slippage)  
- Statistics: cumulative return, annualized return/vol, **Sharpe (√252)**, **max drawdown**, and strategy vs buy-and-hold plot

---

## 🔧 Customize

- **Change the asset**: edit `ticker = 'SPY'` to any liquid symbol supported by Yahoo (e.g., `QQQ`, `IWM`, `AAPL`).  
- **Change horizon**: relabel for multi-day horizons (e.g., `Close[t+k] > Close[t]`) and re-run.  
- **Add features**: macro/sentiment/alt-data, order-flow, higher-frequency bars.  
- **Trading rules**: add transaction costs, slippage, position sizing, risk overlays.

---

## ⚠️ Limitations

- Daily bars + simple features ⇒ modest predictive power (AUC ~0.6–0.7 is common).  
- Backtest ignores **fees, slippage, borrow**, and **taxes**.  
- Thresholds and hyperparams can **overfit** without careful walk-forward validation.  
- Performance is **non-stationary**; periodic re-training and monitoring are required.

---

## 📦 Minimal requirements

```
python >= 3.9
pandas, numpy, matplotlib
scikit-learn
xgboost
yfinance
```

---

## 🙏 Acknowledgments

- Yahoo Finance data via **`yfinance`**  
- **XGBoost** for gradient boosting  
- **scikit-learn** utilities for model selection and metrics

---

## 🗺️ Roadmap / Next steps

- Cost-sensitive thresholding & calibration (e.g., expected utility)  
- Add **commission/slippage** and **position sizing** to the backtest  
- Try stacked/ensemble GBMs or sequence models (RNN/Transformer)  
- Broaden features (macro, sentiment, intraday microstructure)
