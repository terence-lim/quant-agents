# How to Use the `Panel` Data Class and API

This guide describes how to use the `Panel` data class and API for synthesized code and data-panel workflows.

## Panel coding conventions for synthesized code

- Prefer `from qrafti import Panel, DATES, plt_savefig` when relevant.
- Load existing panels with `Panel().load(panel_id, **DATES)`.
- Use Panel methods (`apply`, `trend`, `restrict`, operators) for panel operations; use `.frame` for pandas/lib usage.
- If plotting, use `plt_savefig()` to save the figure and include the image filename in output JSON.
- At the end of data-panel scripts, always persist and print `as_payload()` in JSON form.

## Mental model

- `Panel` wraps a pandas object and is primarily used as a single-column panel over index levels:
  - 2-level MultiIndex: `(date, stock)` (main case)
  - 1-level date index
  - scalar / empty edge cases
- Use `.frame` to access the underlying pandas object.
- Most panel methods return a new `Panel`, enabling method chaining.

## How to initialize from values

- Empty panel: `Panel()`
- Scalar value: `Panel(0.0)`, `Panel(1)`, `Panel(True)`
- From pandas DataFrame: `Panel(df)` where index names are date/stock-compatible.
- From pandas Series: `Panel(series)` (converted to single-column DataFrame).
- From existing panel: `Panel(existing_panel)` (copy semantics).

## Minimal Panel cheatsheet

- Initialize: `Panel()`, `Panel(scalar)`, `Panel(df)`, `Panel(series)`, `Panel(other_panel)`.
- Cross-section by date: `panel.apply(helper, reference=None, how="left", fill_value=0)`.
- Time-series by stock: `panel.trend(helper, reference=None|list[Panel], how="left", fill_value=0)`.
- Useful APIs: `copy`, `ones_like`, `shift`, `restrict`, `.frame`, `as_payload`.
- Operators: arithmetic/comparison/logical (`+ - * / **`, `== != < <= > >=`, `& |`), unary (`-`, `~`, `abs`, `log1p`, `exp`, `expm1`), and `@` (date-wise dot product).

## Lifecycle pattern (recommended in scripts)

1. `Panel().load(panel_id, **DATES)` to retrieve persisted data.
2. Transform using panel methods (`apply`, `trend`, math ops, etc.) or pandas via `.frame`.
3. At the end of data-panel scripts, persist and print `as_payload()` in JSON form so the caller receives the resulting panel id of persisted output.

## Core APIs you will use often

- `load(name, start_date=None, end_date=None)` → load cached frame.
- `save(name="")` → persist and set panel id (`name`).
- `as_payload()` → `{"results_panel_id": ..., "nlevels": ..., "rows": ...}`.
- `frame` → underlying pandas DataFrame/scalar.
- `apply(func, reference=None, how="left", fill_value=0, **kwargs)`:
  - Cross-sectional by date (groupby level 0).
  - Helper receives a DataFrame for one date across stocks.
- `trend(func, reference=None|list[Panel], how="left", fill_value=0, interval=0, **kwargs)`:
  - Time-series by stock (groupby level 1).
  - Helper receives a DataFrame/Series for one stock across time.
- Common transforms: `shift`, `log1p`, `exp`, `expm1`, arithmetic/comparison operators.

## Additional APIs often used in pipelines

- `copy()` → clone panel data + metadata.
- `ones_like()` → same index/shape but values set to `1`.
- `shift(shift=1)` → date relabel shift via calendar mapping (not rolling/resampling).
- `restrict(...)` → filter rows by date bounds, value bounds, boolean `mask`, `subset` index, and optional minimum stocks per date.

## Alignment behavior (`apply` / `trend`)

- Optional `reference` panel(s) are joined with the main panel before helper execution.
- `how` controls join semantics (`left`, `inner`, `right`, `outer` where applicable).
- `fill_value` fills missing values after join.
- `apply`/`trend` typically return first output column as new panel values.

## Operators quick reference

- Binary arithmetic: `+`, `-`, `*`, `/`, `**` (and reverse versions).
- Binary comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`.
- Binary logical: `&`, `|`.
- By-group dot product: `@` (date-wise dot product).
- Unary operators/methods: unary negation `-panel`, logical invert `~panel`, `abs()`, `exp()`, `log1p()`, `expm1()`.

All binary operators align panel indexes internally; missing matches are handled with method-defined join/fill behavior.

## Reference examples to adapt when synthesizing code

### Example 1 — `.frame` + matplotlib plotting

```python
from qrafti import Panel, DATES, plt_savefig
import matplotlib.pyplot as plt
import json

panel_id = "HML"
returns_panel = Panel().load(panel_id, **DATES)
returns_df = returns_panel.frame

plt.plot(returns_df.index, returns_df.cumsum().values)
plt.title(panel_id)
plt.xlabel("Date")
plt.ylabel("Return")
plt.tight_layout()

out_dict = {**returns_panel.as_payload(), "image file name": plt_savefig()}
print(json.dumps(out_dict))
````

### Example 2 — Cross-sectional `apply()` winsorization by date

```python
from qrafti import Panel, DATES
import pandas as pd
import json

def winsorize_helper(x, lower=0.05, upper=0.95):
    if x.shape[1] > 1:
        lo, hi = x.loc[x.iloc[:, -1].astype(bool)].iloc[:, 0].quantile([lower, upper]).values
    else:
        lo, hi = x.iloc[:, 0].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=lo, upper=hi)

data_panel = Panel().load("RET", **DATES)
indicator_panel = (Panel().load("EXCHCD", **DATES) == 1)
result_panel = data_panel.apply(winsorize_helper, indicator_panel, how="left", fill_value=0)
print(json.dumps(result_panel.as_payload()))
```

### Example 3 — Time-series rolling metric with `trend()`

```python
from qrafti import Panel, DATES
import pandas as pd
import json

def rolling_helper(df: pd.DataFrame) -> pd.Series:
    window, skip = 12, 1
    return df.shift(periods=skip).rolling(window=window - skip).sum().where(df.notna())

log_returns = Panel().load("RET", **DATES).log1p()
result_panel = log_returns.trend(rolling_helper)
print(json.dumps(result_panel.as_payload()))
```

## Output rules

* Return concise results including:

  * executed code (when synthesized), and
  * raw stdout / structured error from execute_python.
* Preserve JSON outputs exactly when present.

