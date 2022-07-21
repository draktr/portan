import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self, tickers, weights, start, end, risk_free_rate=0.01):
        self.tickers=tickers
        self.start=start
        self.end=end
        # ? self.weights = weights
        self.risk_free_rate = risk_free_rate

        self.securities_returns = pd.DataFrame(columns=self.tickers)
        self.prices = pd.DataFrame(columns=self.tickers)
        for ticker in self.tickers:
            price_current = yf.download(ticker, start=self.start, end=self.end)
            self.prices[ticker] = price_current["Adj Close"]
            returns_current = self.prices["Adj Close"].pct_change()
            self.securities_returns[ticker] = returns_current

        #returns = self.securities_returns.to_numpy()
        self.portfolio_returns = np.dot(self.securities_returns.to_numpy(), weights)
        #TODO: if-else about importing .csv data, periods etc

    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def returns_with_dates(self):
        # returns pandas dataframe
        returns_with_dates = pd.DataFrame(columns=["Returns"], index=self.securities_returns.index)
        returns_with_dates["Returns"] = self.portfolio_returns
        return returns_with_dates

    def portfolio_daily_volatility(self):
        volatility = np.std(self.portfolio_returns)
        return volatility

    def portfolio_cumulative_returns(self):
        portfolio_returns = self.returns_with_dates()
        portfolio_cumulative_returns = (portfolio_returns["Returns"] + 1).cumprod()
        return portfolio_cumulative_returns

    def plot_portfolio_returns(self):
        portfolio_returns = self.returns_with_dates()

        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns["Returns"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        mpl.show()

    def plot_portfolio_returns_distribution(self):
        portfolio_returns = self.returns_with_dates()

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Portfolio Returns Distribution")
        mpl.show()

    def plot_portfolio_cumulative_returns(self):
        portfolio_cumulative_returns = self.portfolio_cumulative_returns()

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_cumulative_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("Portfolio Cumulative Returns")
        mpl.show()

    def average_daily_return(self):
        returns = self.returns_with_dates()
        average_return = returns.mean()
        return average_return

    def sharpe(self):
        mean_daily_returns = np.mean(self.portfolio_returns)
        volatility = np.std(self.portfolio_returns)
        sharpe_ratio = (100*mean_daily_returns - self.risk_free_rate)/volatility
        return sharpe_ratio

    def downside_deviation(self):
        portfolio_returns = self.portfolio_returns - self.risk_free_rate
        portfolio_returns = portfolio_returns[portfolio_returns<0]
        downside_deviation = np.std(portfolio_returns)
        return downside_deviation

    def sortino(self):
        mean_daily_returns = np.mean(self.portfolio_returns)
        downside_volatility = self.downside_deviation(self.risk_free_rate)
        sortino_ratio = (100*mean_daily_returns - self.risk_free_rate)/downside_volatility
        return sortino_ratio


# TODO: same analytics for each security in the portfolio separately
# TODO: other ratios, maximum drawdown; martin ratio
# TODO: ulcer index adn other measurements of pain
# TODO: do portfolio metrics in one method
# TODO: separate one for checking which weigths
# TODO: asset covariance or correlation matrix
