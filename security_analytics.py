import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as  mpl


class SecurityAnalytics():

    def __init__(self, ticker, period):
        self.prices = yf.download(ticker, period)

    def security_return(self, ticker, period):
        final_price=self.prices["Adj Close"][-1]
        initial_price=self.prices["Adj Close"][0]
        security_return=(final_price-initial_price)/initial_price
        return security_return

    def security_daily_returns(self):
        daily_returns = self.prices["Adj Close"].pct_change()
        daily_returns.to_numpy()
        return daily_returns

    def security_monthly_returns(self):
        monthly_returns = self.prices["Adj Close"].resample("M").ffill().pct_change()
        monthly_returns.to_numpy()
        return monthly_returns

    def security_daily_volatility(self):
        daily_returns = self.security_daily_returns()
        std=np.std(daily_returns)
        return std

    def security_cumulative_returns(self):
        security_returns = self.security_daily_returns()
        security_cumulative_returns = (security_returns + 1).cumprod()
        return security_cumulative_returns

    def plot_security_returns(self):
        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.plot(self.security_daily_returns())
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("security Daily Returns")
        mpl.show()

    def plot_security_returns_distribution(self):
        security_returns = self.security_daily_returns(security_daily_returns)

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        security_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("security Returns Distribution")
        mpl.show()

    def plot_security_cumulative_returns(self):
        security_cumulative_returns = self.security_cumulative_returns()

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        security_cumulative_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("security Cumulative Returns")
        mpl.show()
