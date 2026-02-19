"""
Please provide a detailed plan including tool calls to perform
the following analysis, but do not execute the plan yet:

Construct book equity as stockholder equity (seq) minus the book value of preferred stock,
where preferred stock is defined as the redemption (pstkrv), liquidation
(pstkl), or par value (pstk), if available, in that order.

Each December, divide book equity of a firm’s fiscal year-end at or
before December by the company market capitalization (CAPCO) at the
end of the year. Then construct book-to-market equity in June as the lagged values
from the previous December.

Sort stocks independently on market equity (CAP) into small and big
stocks using the median market capitalization of all stocks traded on
the NYSE as breakpoints, and on book-to-market equity into growth,
neutral, and value stocks using the 30th and 70th percentiles of
book-to-market equity of all stocks traded on the NYSE as breakpoints.

Form portfolios as the intersection of these sorts. The portfolios are
weighted by market equity (CAP).  The HML factor portfolio is the
average of a small and a big value portfolio minus the average of a
small and a big growth portfolio in each month. 

Finally, create a scatter plot of the factor returns against its
benchmark returns (with panel ID 'HML').


Please reflect on the plan: identify any potential weaknesses,
inefficiencies or errors, then provide an improved detailed plan.

Great, please execute the revised plan.

"""

dates = DATES
years = Panel().load('YEARS', **dates)
bench_id = 'HML'

# Compute Book Value
pstkrv = Panel().load("pstkrv", **dates)
pstkl = Panel().load("pstkl", **dates)
pstk = Panel().load("pstk", **dates)
seq = Panel().load("seq", **dates)  # total shareholders' equity
preferred_stock = characteristics_coalesce(pstkrv, pstkl, pstk, replace=[0])
#txditc = Panel().load("txditc", **dates).restrict(end_date="1993-12-31")

# less preferred stock, add deferred tax before 1993
book_value = (seq - preferred_stock)  # + txditc)

# positive book values, age 2 years
# book_value = book_value.restrict(min_value=1e-6, **dates) #mask=(years >= 2), 

# Compute Book to Market at December samples
month = 12  # Decembers
book_samples = characteristics_resample(book_value, month=month)
company_value = Panel().load("CAPCO", **dates)
company_value = characteristics_resample(company_value, ffill=True, month=[month])

# Lag Book to Market, restrict to universe and form terciles based on NYSE stocks
lags = 6
char_pf = (book_samples / company_value).shift(lags)

#universe = Panel().load('SHRCD', **dates).apply(pd.DataFrame.isin, values=[10, 11])
#char_pf = char_pf.restrict(mask=universe)
        
# Form size and bm quantiles based on NYSE stocks
nyse = Panel().load("EXCHCD", **dates) == 1
char_quantiles = char_pf.apply(digitize, reference=nyse, cuts=[0.3, 0.7])
    
size_quantiles = Panel().load("CAP", **dates)
size_quantiles = size_quantiles.apply(digitize, nyse, cuts=2)

# Form intersection
market_value = Panel().load("CAP", **dates)
BL = market_value.apply(portfolio_weights, reference=(size_quantiles == 2) & (char_quantiles == 1))
BH = market_value.apply(portfolio_weights, reference=(size_quantiles == 2) & (char_quantiles == 3))
SL = market_value.apply(portfolio_weights, reference=(size_quantiles == 1) & (char_quantiles == 1))
SH = market_value.apply(portfolio_weights, reference=(size_quantiles == 1) & (char_quantiles == 3))
composite_portfolio = (SH - SL + BH - BL) / 2

composite_returns = portfolio_returns(composite_portfolio)
bench = Panel().load(bench_id, **dates)  #.restrict(subset=composite_returns)
composite_returns.plot(bench, kind="scatter", title=f"Composite BM vs {bench_id}")

# sort by greatest difference between bench and composite returns
diff = (composite_returns - bench).frame.sort_values(by=0, ascending=False)
print(diff)
print(
    f"Mean absolute difference: {diff.abs().mean().item():.4f}, {diff.mean().item():.4f}, "
    f"{np.corrcoef(composite_returns.frame.values, bench.frame.values, rowvar=False)[0, 1].item():.4f}, "
    f"{composite_returns.frame.abs().mean().item():.4f}, {bench.frame.abs().mean().item():.4f}"
)



"""
Please provide a detailed plan including tool calls to perform the following analysis, 
but do not execute the plan yet:

Please reflect on the plan: identify any potential weaknesses, inefficiencies or errors, 
then provide an improved plan in detail as necessary

Book equity is defined as stockholder equity (seq), plus balance sheet
deferred taxes and investment tax credit (txditc) if available, minus
the book value of preferred stock. The book value of preferred stock
is the redemption (pstkrv), liquidation (pstkl), or par value (pstk),
if available, in that order.

The original HML portfolio is the average return of a small and a big
value portfolio minus the average return of a small and a big growth
portfolio in each month. Formally, HML = 1/2 (Small Value + Big Value)
- 1/2 (Small Growth + Big Growth). Stocks are sorted into six
portfolios by independently sorting them on market equity into small
and big stocks using the median market capitalization of all stocks
traded on the NYSE as breakpoint and by independently sorting them on
book-to-market equity into value, neutral, and growth stocks using the
30th and 70th percentiles of book-to-market equity of all stocks
traded on the NYSE as breakpoints. The portfolios are constructed at
the end of June of year t and are held from July of year t to June of
year t+1. Market equity observed at the end of June of year t is used
to sort stocks on size. The book equity of a firm’s last fiscal year
with fiscal year-end before the end of December of year t-1 divided by
market equity at the end of December of year t-1 is used to sort
stocks on value. The two neutral portfolios (Small Neutral and Big
Neutral) are not used. The six portfolios are value weighted.

"""

    if False:    # DO NOT DELETE -- FF HML replication
        universe = Panel().load('SHRCD', **dates).apply(pd.DataFrame.isin, values=[10, 11])
        years = Panel().load('YEARS', **dates)

        if True: # FF RMW quarterly
            bench_id = 'RMW'
            # Numerator ope
            ope = Panel().load('oibdpq', **dates) - Panel().load('xintq')
            ope = ope.trend(rolling, window=4, agg="sum", interval=3)
#            ope = ope.restrict(mask=(years >= 2), **dates)

            # Denominator be
            pstkq = Panel().load("pstkq", **dates)
            seqq = Panel().load("seqq", **dates)  # total shareholders' equity
            book_value = seqq - pstkq

            # positive book values, age 2 years
            book_value = book_value.restrict(min_value=1e-6, **dates)
#            book_value = book_value.restrict(min_value=1e-6, mask=(years >= 2), **dates)

            month = 12  # Decembers
            lags = 6    # lag 6 months to June
            panel = characteristics_resample(ope/book_value, month=month).shift(lags)
            char_pf = panel.restrict(mask=universe)

        if False: # FF RMW
            bench_id = 'RMW'
            # Numerator ope
            operating_expenses = characteristics_coalesce(
                Panel().load('xopr', **dates),
                Panel().load('cogs', **dates) + Panel().load('xsga', **dates)
            )
            sales = characteristics_coalesce(
                Panel().load('sale', **dates),
                Panel().load('revt', **dates)
            )
            gross_profit = characteristics_coalesce(
                Panel().load('gp', **dates),
                sales - Panel().load('cogs', **dates)
            )
            operating_income_before_depreciation = characteristics_coalesce(
                Panel().load('ebitda', **dates),
                Panel().load('oibdp', **dates),
                sales - operating_expenses,
                gross_profit - Panel().load('xsga', **dates)
            )
            ope = operating_income_before_depreciation - Panel().load('xint')
#            ope = ope.restrict(mask=(years >= 2), **dates)

            # Denominator be
            pstkrv = Panel().load("pstkrv", **dates)
            pstkl = Panel().load("pstkl", **dates)
            pstk = Panel().load("pstk", **dates)
            seq = Panel().load("seq", **dates)  # total shareholders' equity
            preferred_stock = characteristics_coalesce(pstkrv, pstkl, pstk, replace=0)
            txditc = Panel().load("txditc", **dates).restrict(end_date="1993-12-31")
            book_value = (seq - preferred_stock + txditc)

            # positive book values, age 2 years
            book_value = book_value.restrict(min_value=1e-6, **dates)
#            book_value = book_value.restrict(min_value=1e-6, mask=(years >= 2), **dates)

            month = 12  # Decembers
            lags = 6    # lag 6 months to June
            panel = characteristics_resample(ope/book_value, month=month).shift(lags)
            char_pf = panel.restrict(mask=universe)

        if False:  # CMA = pct_change in total assets
            bench_id = 'CMA'
            panel = Panel().load('at', **dates)
            panel1 = (Panel().load("seq", **dates) + Panel().load("dltt", **dates) +
                      Panel().load("lct", **dates)  + Panel().load("lo", **dates)  + 
                      Panel().load("txditc", **dates))
            panel = characteristics_coalesce(panel, panel1, replace=0)

            # 12-month change in total assets
            panel = -panel.trend(pd.DataFrame.pct_change, periods=1, interval=12)
#            panel = panel.restrict(mask=(years >= 2), **dates)

            # December samples
            month = 12  # Decembers
            lags = 6    # lag 6 months to June
            chg_at = characteristics_resample(panel, month=month).shift(lags)
            char_pf = chg_at.restrict(mask=universe)

        if False: # HML
            bench_id = 'HML'
            # Compute Book Value
            pstkrv = Panel().load("pstkrv", **dates)
            pstkl = Panel().load("pstkl", **dates)
            pstk = Panel().load("pstk", **dates)
            seq = Panel().load("seq", **dates)  # total shareholders' equity
            preferred_stock = characteristics_coalesce(pstkrv, pstkl, pstk, replace=0)
            txditc = Panel().load("txditc", **dates).restrict(end_date="1993-12-31")

            # less preferred stock, add deferred tax before 1993
            book_value = (seq - preferred_stock + txditc)

            # positive book values, age 2 years
            book_value = book_value.restrict(min_value=1e-6, **dates)# mask=(years >= 2), 

            # Compute Book to Market at December samples
            month = 12  # Decembers
            book_samples = characteristics_resample(book_value, month=month)
            company_value = Panel().load("CAPCO", **dates)
            company_value = characteristics_resample(company_value, month=month)

            # Lag Book to Market, restrict to universe and form terciles based on NYSE stocks
            lags = 6
            book_market = (book_samples / company_value).shift(lags)
            char_pf = book_market.restrict(mask=universe)
        
        # Form size and bm quantiles based on NYSE stocks
        nyse = Panel().load("EXCHCD", **dates) == 1
        #    nyse = nyse.restrict(mask=universe)
        char_quantiles = char_pf.apply(digitize, reference=nyse, cuts=[0.3, 0.7])
    
        size_quantiles = Panel().load("CAP", **dates)
        #    size_quantiles = size_quantiles.restrict(mask=universe)
        size_quantiles = size_quantiles.apply(digitize, nyse, cuts=2)
        #    size_quantiles = (Panel().load('SIZE_DECILE') > 5) + 1

        # Form intersection
        market_value = Panel().load("CAP", **dates)
        BL = market_value.apply(
            portfolio_weights,
            reference=(size_quantiles == 2) & (char_quantiles == 1),
        )
        BH = market_value.apply(
            portfolio_weights,
            reference=(size_quantiles == 2) & (char_quantiles == 3),
        )
        SL = market_value.apply(
            portfolio_weights,
            reference=(size_quantiles == 1) & (char_quantiles == 1),
        )
        SH = market_value.apply(
            portfolio_weights,
            reference=(size_quantiles == 1) & (char_quantiles == 3),
        )
        composite_portfolio = (SH - SL + BH - BL) / 2

        #    drifted = portfolio_impute(composite_portfolio, drifted=True)
        #    turnover = composite_portfolio - drifted
        #    print(f"Average Turnover: {turnover.apply(np.abs).apply(np.sum, axis=0).apply(np.mean).frame}")

        composite_returns = portfolio_returns(composite_portfolio)
        summary = returns_metrics(composite_returns)
        print(f"Composite Portfolio Summary: {summary}")

        #    composite_returns = composite_returns.restrict(start_date='1972-01-01')
        bench = Panel().load(bench_id, **dates).restrict(subset=composite_returns)
        print(returns_metrics(bench))
        composite_returns.plot(bench, kind="scatter", title=f"Composite BM vs {bench_id}")

        # sort by greatest difference between bench and composite returns
        diff = (composite_returns - bench).frame.sort_values(by=0, ascending=False)
        print(diff)
        print(
            f"Mean absolute difference: {diff.abs().mean().item():.4f}, {diff.mean().item():.4f}, "
            f"{np.corrcoef(composite_returns.frame.values, bench.frame.values, rowvar=False)[0, 1].item():.4f}, "
            f"{composite_returns.frame.abs().mean().item():.4f}, {bench.frame.abs().mean().item():.4f}"
        )
