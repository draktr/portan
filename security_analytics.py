import numpy as np
import pandas as pd
import yfinance as yf

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
        return daily_returns

    def security_monthly_returns(self):
        monthly_returns = self.prices["Adj Close"].resample("M").ffill().pct_change()
        return monthly_returns

    def security_daily_volatility(self):
        daily_returns = self.security_daily_returns()
        std=np.std(daily_returns)
        return std
