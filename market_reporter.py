import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader as pdr

class MarketReporter():
    def __init__(self) -> None:
        self.today=datetime.today()
        self.yesterday=self.today-timedelta(days=1)
        self.two_days_ago=self.today-timedelta(days=2)
        self.week_ago=self.today-timedelta(days=7)

    def majors(self):
        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X", "USDSGD=X", "USDGBP=X" "^RUT", "^FTSE", "^STOXX50E", "^GDAXI", "^HSI", "^STI", "000001.SS", "399001.SZ"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        names=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR", "USDSGD", "USDGBP", "Russell 2000", "FTSE 100", "Euro STOXX 50", "DAX 40", "Hang Seng Index", "The Straits Times Index", "Shanghai Composite", "Shenzhen Index"]
        current_quotes.reindex(names)

        return current_quotes

    def yield_curve_us(self, date="2022-06-30", save=False, show=True):
        treasuries = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
        yields = pdr.DataReader(treasuries, "fred")
        yields = yields.reindex_axis(labels=["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", "5-year", "7-year", "10-year", "20-year", "30-year"], axis=1)

        print(yields)

        yields.loc[date].plot(label=date)
        if save is True:
            mpl.savefig("yield_curve.png", dpi=300)
        if show is True:
            mpl.show()
