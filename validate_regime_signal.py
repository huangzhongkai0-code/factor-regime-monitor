"""Validate whether regime labels contain information about subsequent returns.

This is descriptive validation, not a trading backtest.  States are formed only
from information available on each date; future 1/3/6-month returns are then
grouped by the state observed on that date.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation_outputs"
OUT.mkdir(exist_ok=True)

TICKERS = ["SPY", "IVE", "IVW", "MTUM", "USMV", "IWM"]
STATE_ORDER = ["Contraction", "Neutral", "Expansion"]

raw = yf.download(TICKERS, start="2014-01-01", end="2026-08-28", auto_adjust=True,
                  progress=False, group_by="column")
prices = raw["Close"].dropna(how="all").ffill()
daily = prices.pct_change(fill_method=None)

rolling_return = (1 + daily).rolling(63).apply(np.prod, raw=True) - 1
rolling_mean = rolling_return.rolling(252).mean()
rolling_std = rolling_return.rolling(252).std(ddof=1)
zscore = (rolling_return - rolling_mean) / rolling_std
states = zscore.map(lambda x: "Expansion" if x > 1 else ("Contraction" if x < -1 else "Neutral"))

records = []
for ticker in TICKERS:
    for horizon, label in [(21, "1M"), (63, "3M"), (126, "6M")]:
        forward = prices[ticker].shift(-horizon) / prices[ticker] - 1
        frame = pd.DataFrame({"state": states[ticker], "forward": forward}).dropna()
        for state in STATE_ORDER:
            sample = frame.loc[frame["state"] == state, "forward"]
            records.append({
                "ticker": ticker, "horizon": label, "state": state,
                "observations": int(sample.size), "mean_forward_return": float(sample.mean()),
                "median_forward_return": float(sample.median()),
                "positive_return_rate": float((sample > 0).mean()),
            })
summary = pd.DataFrame(records)
summary.to_csv(OUT / "forward_return_by_state.csv", index=False)

# Aggregate transition matrix across factor series using non-overlapping weekly observations.
transition_counts = pd.DataFrame(0, index=STATE_ORDER, columns=STATE_ORDER, dtype=int)
for ticker in TICKERS:
    weekly = states[ticker].resample("W-FRI").last().dropna()
    for current, nxt in zip(weekly.iloc[:-1], weekly.iloc[1:]):
        transition_counts.loc[current, nxt] += 1
transition = transition_counts.div(transition_counts.sum(axis=1), axis=0)
transition.to_csv(OUT / "weekly_state_transition_matrix.csv")

# Portfolio-level visual: average across the six factor proxies.
plot_data = (summary.groupby(["horizon", "state"])["mean_forward_return"].mean()
             .unstack("state").reindex(index=["1M", "3M", "6M"], columns=STATE_ORDER))
ax = (plot_data * 100).plot(kind="bar", figsize=(9, 5.2),
                            color=["#c00000", "#a5a5a5", "#70ad47"])
ax.set_title("Average Subsequent Return by Observed Regime")
ax.set_ylabel("Mean forward return (%)")
ax.set_xlabel("Forward horizon")
ax.axhline(0, color="#404040", linewidth=.8)
ax.grid(axis="y", alpha=.2)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "forward_return_validation.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(7.4, 5.8))
im = ax.imshow(transition.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3), STATE_ORDER)
ax.set_yticks(range(3), STATE_ORDER)
ax.set_xlabel("Next weekly state")
ax.set_ylabel("Current weekly state")
ax.set_title("Weekly Regime Transition Probabilities")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{transition.iloc[i, j]:.1%}", ha="center", va="center",
                color="white" if transition.iloc[i, j] > .55 else "#17365d", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
fig.tight_layout()
fig.savefig(OUT / "state_transition_matrix.png", dpi=180)
plt.close(fig)

print(plot_data.to_string(float_format=lambda x: f"{x:.2%}"))
print("\nTransition matrix:\n", transition.to_string(float_format=lambda x: f"{x:.1%}"))

