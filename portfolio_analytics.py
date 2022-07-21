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
        volatility = np.std(portfolio_returns)
        return volatility

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

    def average_daily_return(self, weights):
        returns = self.returns_with_dates(weights)
        average_return = returns.mean()
        return average_return

    def sharpe(self, weights, risk_free_rate):
        portfolio_returns = self.portfolio_returns(weights)
        mean_daily_returns = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = (100*mean_daily_returns - risk_free_rate)/volatility
        return sharpe_ratio

    def downside_deviation(self, weights, risk_free_rate):
        portfolio_returns = self.portfolio_returns(weights)
        portfolio_returns = portfolio_returns - risk_free_rate
        portfolio_returns = portfolio_returns[portfolio_returns<0]
        downside_deviation = np.std(portfolio_returns)
        return downside_deviation

    def sortino(self, weights, risk_free_rate):
        portfolio_returns = self.portfolio_returns(weights)
        mean_daily_returns = np.mean(portfolio_returns)
        downside_volatility = self.downside_deviation(weights, risk_free_rate)
        sortino_ratio = (100*mean_daily_returns - risk_free_rate)/downside_volatility
        return sortino_ratio


# TODO: same analytics for each security in the portfolio separately
# TODO: other ratios, maximum drawdown
# TODO: ulcer index adn other measurements of pain
# TODO: do portfolio metrics in one method
# TODO: calculate returns when initiating
# TODO: separate one for checking which weigths
# TODO: asset covariance or correlation matrix
