import numpy as np
import pandas as pd
import yfinance as yf


class PortfolioAnalytics():
    def _init_(self, tickers, data, period):
        self.tickers=tickers
        self.data=data
        self.period=period

    def list_securities(self):
        for ticker in self.tickers
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def all_securities_returns(self):
        all_securities_returns = pd.DataFrame(columns=self.tickers)
        for ticker in self.tickers:
            prices = yf.download(ticker, self.period)
            daily_returns = prices["Adj Close"].pct_change()
            all_securities_returns[ticker] = daily_returns
        return all_securities_returns

    def portfolio_returns(self, weights):
        returns = self.all_securities_returns
        returns_array = returns.to_numpy()
        portfolio_returns = np.dot(returns_array, weights)
        return portfolio_returns

    def portfolio_daily_volatility(self, weights):
        portfolio_returns = self.portfolio_returns(weights)
        std = np.std(portfolio_returns)
        return std


# TODO: plot of daily, monthly returns
# TODO: test


