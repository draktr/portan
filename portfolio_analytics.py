import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self, tickers, start, end):
        self.tickers=tickers
        self.start=start
        self.end=end

        self.all_securities_returns = pd.DataFrame(columns=self.tickers)
        for ticker in self.tickers:
            self.prices = yf.download(ticker, start=self.start, end=self.end)
            daily_returns = self.prices["Adj Close"].pct_change()
            self.all_securities_returns[ticker] = daily_returns
        #TODO: if-else about importing .csv data, periods etc

    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def returns_with_dates(self, weights):
        # returns pandas dataframe
        returns_with_dates = pd.DataFrame(columns=["Returns"], index=self.all_securities_returns.index)
        returns_with_dates["Returns"] = self.portfolio_returns(weights)
        return returns_with_dates

    def portfolio_returns(self, weights):
        # returns numpy array
        returns = self.all_securities_returns.to_numpy()
        portfolio_returns = np.dot(returns, weights)
        return portfolio_returns

    def portfolio_daily_volatility(self, weights):
        portfolio_returns = self.portfolio_returns(weights)
        std = np.std(portfolio_returns)
        return std

    def portfolio_cumulative_returns(self, weights):
        portfolio_returns = self.returns_with_dates(weights)
        portfolio_cumulative_returns = (portfolio_returns["Returns"] + 1).cumprod()
        return portfolio_cumulative_returns

    def plot_portfolio_returns(self, weights):
        portfolio_returns = self.returns_with_dates(weights)

        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns["Returns"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        mpl.show()

    def plot_portfolio_returns_distribution(self, weights):
        portfolio_returns = self.returns_with_dates(weights)

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily Returns")
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
# TODO: add dates for plot methods
# TODO: sharpe, sortio and other ratios
# TODO: ulcer index adn other measurements of pain

