if __name__ == "__main__":
    if False: # True:  # Calculate diff quarterly
        panel = Panel().load('atq')
        diff = panel.trend(pd.DataFrame.diff, periods=1, interval=3)
        pctchange = panel.trend(pd.DataFrame.pct_change, periods=4, interval=3)
        annual = panel.trend(rolling, window=4, agg="sum", interval=3)
        panel[10501]
        diff[10501]
        pctchange[10501]

    if False: # True:  # Calculate diff and pctchange: 30 seconds each fpr full panel
        panel = Panel().load('at')
        diff = panel.trend(pd.DataFrame.diff, periods=1, interval=12)
        pctchange = panel.trend(pd.DataFrame.pct_change, periods=1, interval=12)

    if False: # True:  # coalesce quarterly 12-month changes, equal-weighted
        universe = Panel().load('SHRCD', **dates).apply(pd.DataFrame.isin, values=[10, 11])
        lags = 4
        annual = Panel().load("at", **dates).restrict(min_value=1e-6)
        annual = annual.trend(pd.DataFrame.pct_change, interval=12, periods=1)
        quarterly = Panel().load("atq", **dates).restrict(min_value=1e-1)
        quarterly = quarterly.trend(pd.DataFrame.pct_change, interval=3, periods=4)
        factor_pf = -characteristics_coalesce(annual, quarterly, replace=0)

        ### how to make sure these become monthly??
#        factor_pf = characteristics_resample(factor_pf, month=[3, 6, 9, 12])
        factor_pf.restrict(subset=universe).shift(lags)

        nyse_pf = Panel().load("EXCHCD", **dates).apply(pd.DataFrame.isin, values=[1])
        size_pf = Panel().load("CAP", **dates)
        decile_pf = size_pf.apply(digitize, nyse_pf, cuts=10)
        quantiles_pf = factor_pf.apply(digitize, reference=decile_pf > 2, cuts=3)

        # equal-weighted, non-micro stocks
        long_pf = (decile_pf > 2).apply(portfolio_weights, reference=quantiles_pf == 3)
        short_pf = (decile_pf > 2).apply(portfolio_weights, reference=quantiles_pf == 1)
        spreads_pf = long_pf - short_pf
        composite_returns = portfolio_returns(spreads_pf)

        # FF5 appraisal ratio # Panel_appraisal_ratio(model="FF3", "CAPM", "FF5") = annualized % alpha at 1% vol
        reg, resid = returns_regression(
            composite_returns,
            [Panel().load(s) for s in ['Mkt-RF', 'HML', 'SMB', 'CMA', 'RMW']]
        )
        print(np.sqrt(12) * reg['intercept'] / np.sqrt(reg['resid_variance']))
        print(json.dumps(reg, indent=2))

