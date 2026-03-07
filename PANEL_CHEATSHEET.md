# qrafti `Panel` Cheat-Sheet (for Coding Agents)

This cheat-sheet summarizes how `Panel` works in `qrafti.py` and provides ready-to-adapt patterns.

## 1) Mental model

- `Panel` wraps a pandas object and is primarily used as a **single-column panel** over index levels:
  - 2-level MultiIndex: `(date, stock)` (main case)
  - 1-level date index
  - scalar / empty edge cases
- Use `.frame` to access the underlying pandas object.
- Most panel methods return a new `Panel`, enabling method chaining.

## 1.1) How to initialize from values

- Empty panel: `Panel()`
- Scalar value: `Panel(0.0)`, `Panel(1)`, `Panel(True)`
- From pandas DataFrame: `Panel(df)` where index names are date/stock-compatible.
- From pandas Series: `Panel(series)` (converted to single-column DataFrame).
- From existing panel: `Panel(existing_panel)` (copy semantics).

## 2) Lifecycle pattern (recommended in scripts)

1. `Panel().load(panel_id, **DATES)` to retrieve persisted data.
2. Transform using panel methods (`apply`, `trend`, math ops, etc.) or pandas via `.frame`.
3. `result_panel.save()` to persist output.
4. `print(result_panel.as_payload())` (or JSON) so caller receives resulting panel id.

## 3) Core APIs you will use often

- `load(name, start_date=None, end_date=None)` → load cached frame.
- `save(name="")` → persist and set panel id (`name`).
- `as_payload()` → `{"results_panel_id": ..., "nlevels": ..., "rows": ...}`.
- `frame` → underlying pandas DataFrame/scalar.
- `apply(func, reference=None, how="left", fill_value=0, **kwargs)`:
  - Cross-sectional by **date** (groupby level 0).
  - Helper receives a DataFrame for one date across stocks.
- `trend(func, reference=None|list[Panel], how="left", fill_value=0, interval=0, **kwargs)`:
  - Time-series by **stock** (groupby level 1).
  - Helper receives a DataFrame/Series for one stock across time.
- common transforms: `shift`, `log1p`, `exp`, `expm1`, arithmetic/comparison operators.

## 3.1) Additional APIs often used in pipelines

- `copy()` → clone panel data + metadata.
- `ones_like()` → same index/shape but values set to `1`.
- `shift(shift=1)` → **date relabel shift** via calendar mapping (not rolling/resampling).
- `restrict(...)` → filter rows by date bounds, value bounds, boolean `mask`, `subset` index, and optional minimum stocks per date.

## 4) Alignment behavior (`apply` / `trend`)

- Optional `reference` panel(s) are joined with the main panel before helper execution.
- `how` controls join semantics (`left`, `inner`, `right`, `outer` where applicable).
- `fill_value` fills missing values after join.
- `apply`/`trend` typically return first output column as new panel values.

## 5) Operators quick reference

- Binary arithmetic: `+`, `-`, `*`, `/`, `**` (and reverse versions).
- Binary comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`.
- Binary logical: `&`, `|`.
- By-group dot product: `@` (date-wise dot product).
- Unary operators/methods: unary negation `-panel`, logical invert `~panel`, `abs()`, `exp()`, `log1p()`, `expm1()`.

All binary operators align panel indexes internally; missing matches are handled with method-defined join/fill behavior.

---

## Example 1 — Use matplotlib on underlying pandas DataFrame (`.frame`)

```python
from qrafti import Panel, DATES, plt_savefig
import matplotlib.pyplot as plt
import json

panel_id = "HML"
returns_panel = Panel().load(panel_id, **DATES)      # Panel
returns_df = returns_panel.frame                      # pandas DataFrame

plt.figure(figsize=(8, 4))
plt.plot(returns_df.index, returns_df.cumsum().values)
plt.title(panel_id)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.tight_layout()
image_file = plt_savefig()

# Persist panel and return payload + image metadata
result_panel = returns_panel.save()
out_dict = {
    **result_panel.as_payload(),
    "image file name": image_file,
}
print(json.dumps(out_dict))
```

## Example 2 — Cross-sectional winsorization with `apply()` (by date across stocks)

```python
from qrafti import Panel, DATES
import pandas as pd
import json


def winsorize_helper(x, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    Winsorize first column by quantiles; if indicator exists in last column,
    compute quantiles only on indicator-True rows.
    """
    if x.shape[1] > 1:
        q_low, q_high = (
            x.loc[x.iloc[:, -1].astype(bool)]
            .iloc[:, 0]
            .quantile([lower, upper])
            .values
        )
    else:
        q_low, q_high = x.iloc[:, 0].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=q_low, upper=q_high)


panel_id = "RET"
data_panel = Panel().load(panel_id, **DATES)

indicator_panel_id = "EXCHCD"
indicator_panel = (Panel().load(indicator_panel_id, **DATES) == 1) if indicator_panel_id else None

result_panel = data_panel.apply(
    winsorize_helper,
    indicator_panel,
    how="left",
    fill_value=0,
).save()

print(json.dumps(result_panel.as_payload()))
```

## Example 3 — Time-series residual regression with `trend()` (by stock across time)

```python
from qrafti import Panel, DATES
import pandas as pd
import numpy as np
import json


def residuals_helper(x: pd.DataFrame) -> pd.Series:
    """Residuals from OLS: y ~ 1 + X."""
    y = x.iloc[:, 0].values
    X = x.iloc[:, 1:].values
    X = np.column_stack([np.ones(len(X)), X])

    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        return pd.Series([np.nan] * len(x), index=x.index)

    try:
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        betas = np.linalg.pinv(X) @ y

    resid = y - X @ betas
    return pd.Series(resid, index=x.index)


returns_panel = Panel().load("RET", **DATES)
factor_panels = [Panel().load(pid, **DATES) for pid in ["Mkt-RF", "SMB", "HML"]]

result_panel = returns_panel.trend(residuals_helper, factor_panels).save()
print(json.dumps(result_panel.as_payload()))
```

## Example 4 — Rolling time-series statistic with `trend()` (by stock over time)

```python
from qrafti import Panel, DATES
import pandas as pd
import json


def rolling_helper(df: pd.DataFrame) -> pd.Series:
    """12m momentum with 1m skip on log returns."""
    window = 12
    skip = 1
    agg = "sum"
    return df.shift(periods=skip).rolling(window=window - skip).agg(agg).where(df.notna())


log_returns = Panel().load("RET", **DATES).log1p()
result_panel = log_returns.trend(rolling_helper).save()
print(json.dumps(result_panel.as_payload()))
```
