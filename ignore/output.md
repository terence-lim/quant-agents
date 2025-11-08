12-1 Momentum: Prior 12 month returns excluding most recent month
------------

You are a sell-side quantitative researcher writing a captivating research memo
on this new financial signal for predicting stock returns. You should also provide a title name for the signal.

Please follow these guidelines for writing the research memo:

1. Motivation (1 paragraph, ~100 words): 
    * Broad statement on market efficiency or asset pricing. 
    * Identify a gap in the current practice and literature.
    * Use active voice and declarative statements.

2. Hypothesis Development (1 paragraph, ~150 words):
    * Present economic mechanisms linking signal to returns.
    * Draw on theoretical frameworks.
    * Support claims with citations.

3. Results Summary (1-2 paragraphs, ~200 words):
    * Lead with the strongest statistical finding.
    * Summarize the key results in a narrative form, including economic significance.
    * Do not merely cite numbers; interpret them.

4. Contribution (1 paragraph, ~150 words):
    * Position relative to 3-4 related finance/accounting journal articles.
    * Highlight methodological innovations.

In your writing, please:

* Use active voice (e.g., “We find”).
* Maintain clarity and conciseness.
* Avoid jargon; explain technical terms.
* Use present tense for established findings.
* Use past tense for specific results.
* Make clear distinctions between correlation and causation.
* Avoid speculation beyond the data.

Output in markdown format with sections: Introduction, Hypothesis Development, Results, Contribution.

Base the results section strictly on the following data, matching its terminology and precision:


### % of Names Covered by Year
|   year |   % of names covered |
|-------:|---------------------:|
|   2020 |               109.16 |
|   2021 |                98.97 |
|   2022 |               108.42 |
|   2023 |               117.87 |
|   2024 |               117.91 |

### % of Market Cap Covered by Year
|   year |   % of cap covered |
|-------:|-------------------:|
|   2020 |              98.95 |
|   2021 |              97.7  |
|   2022 |              99.22 |
|   2023 |              99.66 |
|   2024 |              99.63 |

### Statistics of Tercile Spread Portfolios
(weighted by market cap winsorized at 80th NYSE percentile

|                  |   Annualized Return |   Volatility |   Skewness |   Excess Kurtosis |   Sharpe Ratio |   Max Drawdown |   Number of Observations | Start Date   | End Date   |
|:-----------------|--------------------:|-------------:|-----------:|------------------:|---------------:|---------------:|-------------------------:|:-------------|:-----------|
| high-low tercile |           0.0342855 |     0.175541 |  -0.847221 |           1.48049 |       0.195313 |      -0.335825 |                       59 | 2020-02-29   | 2024-12-31 |

### Alpha, coefficients and t-statistics by Model

| Mean Returns   |   coefficients |   t-stats |
|:---------------|---------------:|----------:|
| intercept      |         0.0029 |    0.4331 |

| CAPM      |   coefficients |   t-stats |
|:----------|---------------:|----------:|
| intercept |         0.0079 |    1.3451 |
| Mkt-RF    |        -0.4635 |   -4.3796 |

| Fama-French 3-Factor Model   |   coefficients |   t-stats |
|:-----------------------------|---------------:|----------:|
| intercept                    |         0.0061 |    1.163  |
| Mkt-RF                       |        -0.34   |   -3.4189 |
| SMB                          |        -0.6876 |   -3.9741 |
| HML                          |        -0.042  |   -0.3759 |

### Alpha and t-statistics by Model and Size Quintile
(lower quintiles have smaller market cap)

| Model         |   Size Quintile 1 |   Size Quintile 2 |   Size Quintile 3 |   Size Quintile 4 |   Size Quintile 5 |
|:--------------|------------------:|------------------:|------------------:|------------------:|------------------:|
| mean          |            0.0069 |            0.0058 |            0.0047 |            0.0048 |            0.0067 |
| t-stat        |            1.557  |            1.1985 |            0.8899 |            0.7896 |            0.7967 |
| alpha (CAPM)  |            0.0088 |            0.0084 |            0.008  |            0.0089 |            0.0119 |
| t-stat (CAPM) |            2.0211 |            1.8158 |            1.6098 |            1.5845 |            1.517  |
| alpha (FF3)   |            0.008  |            0.0074 |            0.0068 |            0.0072 |            0.0093 |
| t-stat (FF3)  |            1.8752 |            1.6622 |            1.4454 |            1.4204 |            1.365  |