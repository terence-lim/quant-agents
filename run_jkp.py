"""
Please provide a detailed plan including tool calls to perform
the following analysis, but do not execute the plan yet.
Reflect on the plan, and identify any potential weaknesses,
inefficiencies or errors, then provide an improved detailed plan:


Please provide a detailed plan including tool calls to perform
the following analysis, but do not execute the plan yet:


Define price momentum characteristic as stocks' past 12 months returns skipping one month.

Sort stocks into characteristic terciles (top/middle/bottom third) with breakpoints based on non-micro stocks,
where micro stocks are all stocks whose market capitalization is below the NYSE 20th percentile.

For each tercile, compute its "capped value" weighted portfolio, meaning that we weight stocks by their market equity
winsorized at the NYSE 80th percentile.  
The factor returns is then defined as the top-tercile portfolio return minus the bottom-tercile portfolio return.

Create a scatter plot of the factor returns against its benchmark returns (Panel ID 'ret_12_1_ret_vw_cap').


Please reflect on the plan: identify any potential weaknesses,
inefficiencies or errors, then provide an improved detailed plan.

Great, please execute the revised plan.

"""

dates = DATES
window, skip = 12, 1
log1p_ret = Panel().load('RET', **dates).log1p()
factor_pf = log1p_ret.trend(rolling, window=window, skip=skip, agg="sum", interval=1).expm1()

#nyse_pf = Panel("crsp_exchcd", **dates).apply(pd.DataFrame.isin, values=[1])
nyse_pf = Panel().load("EXCHCD", **dates).apply(pd.DataFrame.isin, values=[1])
#size_pf = Panel("me", **dates)
size_pf = Panel().load("CAP", **dates)
decile_pf = size_pf.apply(digitize, nyse_pf, cuts=10)
quintile_pf = size_pf.apply(digitize, nyse_pf, cuts=5)
terciles_pf = factor_pf.apply(digitize, reference=decile_pf > 2, cuts=3)

vwcap_pf = size_pf.apply(winsorize, nyse_pf, lower=0, upper=0.80)
long_pf = vwcap_pf.apply(portfolio_weights, reference=terciles_pf == 3)
short_pf = vwcap_pf.apply(portfolio_weights, reference=terciles_pf == 1)
long_ret = portfolio_returns(long_pf)
short_ret = portfolio_returns(short_pf)
spreads_pf = long_pf - short_pf
#composite_returns = portfolio_returns(spreads_pf)
composite_returns = long_ret - short_ret

bench = Panel().load("ret_12_1_ret_vw_cap", **dates)
composite_returns.plot(bench, kind="scatter")


"""
# Definitions
- benchmark returns: The Panel ID for characteristics benchmark returns is the characteristic name followed by the string '_ret_vw_cap'. The characteristic name must be one that you created or was defined in the variables descriptions.
- factor: factor typically means the returns to a factor portfolio; factor returns are constructed from factor portfolio weights by calling suitable tool, while factor portfolio weights are constructed from raw scores or stock characteristics by calling other tools.
- micro stocks are all stocks whose market capitalization is below the NYSE 20th percentile.
- NYSE stocks have crsp exchcd = 1

For each characteristic, we build the one-month-holding-period factor return
within each country as follows. First, in each country and month, we sort stocks
into characteristic terciles (top/middle/bottom third) with breakpoints based
on non-micro stocks in that country. For each tercile, we compute its "capped
value weight" return, meaning that we weight stocks by their market equity
winsorized at the NYSE 80th percentile. This construction ensures that tiny
stocks have tiny weights and any one mega stock does not dominate a portfo-
lio in an effort to create tradable, yet balanced, portfolios. The factor is then
defined as the high-tercile return minus the low-tercile return, corresponding
to the excess return of a long-short zero-net-investment strategy.
Finally, evaluate the performance of the factor portfolio 
and create a scatter plot of the factor against its benchmark returns.


"""
