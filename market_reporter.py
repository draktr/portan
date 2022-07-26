import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import yfinance as yf

class MarketReporter():
    def __init__(self) -> None:
        pass

    def important_ones(self):
        tickers=["^GSPC", "^DJI", "^TNX"]
        for ticker in tickers:
            sec=yf.Ticker(ticker)
            price=sec.info["regularMarketPrice"]
            print(ticker, price)

    def yield_curve(self):
        pass
