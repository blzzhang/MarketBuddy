import datetime as dt
import numpy as np 
import pandas as pd 
import pandas_datareader.data as web 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('ggplot')

def hist_perf(ticker, start, end):
    tickers = [ticker, 'SPY']
    graph = pd.DataFrame
    first = True
    for tick in tickers:
        dataFirst = web.DataReader(tick,'iex',start,end)
        if first:
            graph = dataFirst[['close']]
            graph.rename({'close': tick},axis='columns',inplace=True)
            first = False
        else:
            graph.insert(0, tick, dataFirst['close'],allow_duplicates=True)
        startPrice = float(dataFirst['close'].iloc[0])
        endPrice = float(dataFirst['close'].iloc[-1])
        roi = round(endPrice/startPrice - 1,3)
        percentReturn = "{0:.1f}%".format(roi * 100)
        print("\nThe return of {} was {}.\n".format(tick,percentReturn))
    graph = graph.pct_change()
    graph = graph.multiply(100)
    graph.plot(kind='line',figsize=(20,13),use_index=True)
    plt.ylabel("Daily Return (%)")
    plt.legend(prop={'size':20})
    print('Price Change Chart')
    plt.show()


def beta_hedge(tickers_portfolio, start, end):
    tickers = []
    for key in tickers_portfolio:
        tickers.append(key)
    tickers.append('SPY')
    data = web.DataReader(tickers,'iex',start,end)
    first = True
    for key in data:
        df = data[key]
        if first:
            close = df[['close']]
            close.rename({'close': key},axis='columns',inplace=True)
            first = False
        else:
            close.insert(0,key,df['close'],allow_duplicates=True)
    change = close.pct_change()
    alpha = {}
    beta = {}
    for key in tickers_portfolio:
        results = smf.ols('{} ~ SPY'.format(key), data=change).fit()
        alpha_coef = results.params[0]
        beta_coef = results.params[1]
        alpha[key] = alpha_coef
        beta[key] = beta_coef
    
    #calculate portfolio beta
    portfolio_beta = 0
    for key in beta:
        portfolio_beta = portfolio_beta + (beta[key] * tickers_portfolio[key] / 100)
    print("\nYour total portfolio beta relative to the S&P 500 index is {}.\n".format(portfolio_beta))
    portfolio_return =change.copy()
    
    #calculate pro-rated portfolio daily return
    for key in tickers_portfolio:
        portfolio_return[key] = portfolio_return[key].apply(lambda x: x * tickers_portfolio[key] / 100)
    portfolio_return['Total Return'] = portfolio_return[list(portfolio_return.columns)].sum(axis=1,min_count=1)
    
    #calculate portfolio hedge alpha with risk free rate = 0
    portfolio_hedge = -1*portfolio_beta*change['SPY'] + portfolio_return['Total Return']
    
    change['Portfolio Hedge'] = portfolio_hedge
    change = change.multiply(100)
    change.plot(kind='line',figsize=(20,13),use_index=True)
    plt.ylabel("Daily Return (%)")
    plt.legend(prop={'size':20})
    print('Portfolio Hedge Chart')
    plt.show()
    
    
    

start = dt.datetime(2016,1,1)
end = dt.datetime(2016, 12, 31)

tickers = ['AMZN','GOOG','SLF']
weighted_tickers = {'AMZN': 20, 'GOOG': 40, 'SLF': 40}


#daily_return = get_return(tickers, start, end, log_return=True)

#daily_return.head()

#hist_perf('AAPL',dt.datetime(2016,1,1),dt.datetime(2016,12,31))
beta_hedge(weighted_tickers,start,end)

#if __name__=="__main__":
