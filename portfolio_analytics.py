import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self, tickers, period):
        self.tickers=tickers
        self.period=period

    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def all_securities_returns(self):
        all_securities_returns = pd.DataFrame(columns=self.tickers)
        for ticker in self.tickers:
            prices = yf.download(ticker, self.period)
            daily_returns = prices["Adj Close"].pct_change()
            all_securities_returns[ticker] = daily_returns
        all_securities_returns.to_numpy()
        return all_securities_returns

    def portfolio_returns(self, weights):
        returns = self.all_securities_returns
        portfolio_returns = np.dot(returns, weights)
        return portfolio_returns

    def portfolio_daily_volatility(self, weights):
        portfolio_returns = self.portfolio_returns(weights)
        std = np.std(portfolio_returns)
        return std

    def portfolio_cumulative_returns(self, weights):
        portfolio_returns = self.portfolio_returns(weights)
        portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()
        return portfolio_cumulative_returns

    def plot_portfolio_returns(self, weights):
        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.plot(self.portfolio_returns(weights))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        mpl.show()

    def plot_portfolio_returns_distribution(self, weights):
        portfolio_returns = self.portfolio_returns(weights)

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Portfolio Returns Distribution")
        mpl.show()

    def plot_portfolio_cumulative_returns(self, weights):
        portfolio_cumulative_returns = self.portfolio_cumulative_returns(weights)

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_cumulative_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("Portfolio Cumulative Returns")
        mpl.show()


# TODO: same analytics for each security in the portfolio separately
