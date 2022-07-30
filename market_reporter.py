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

    def majors(self): # TODO: make a shorter version
        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X", "USDSGD=X", "USDGBP=X" "^RUT", "^FTSE", "^STOXX50E", "^GDAXI", "^HSI", "^STI", "000001.SS", "399001.SZ"]
        current_quotes = pd.Series(index=tickers)

        #TODO: dont use for loop
        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        names=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR", "USDSGD", "USDGBP", "Russell 2000", "FTSE 100", "Euro STOXX 50", "DAX 40", "Hang Seng Index", "The Straits Times Index", "Shanghai Composite", "Shenzhen Index"]
        current_quotes.reindex(names)

        return current_quotes

    def yield_curve_us(self, date="2022-06-30", show=True, save=False):
        treasuries = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
        yields = pdr.DataReader(treasuries, "fred")
        yields = yields.reindex_axis(labels=["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", "5-year", "7-year", "10-year", "20-year", "30-year"], axis=1)

        yields.loc[date].plot(label=date)
        if save is True:
            mpl.savefig("yield_curve_us.png", dpi=300)
        if show is True:
            mpl.show()

        return yields

    def yield_curve_euro(self, date="2022-06-01", show=True, save=False):
        raw_data = pdr.DataReader("teimf060", "eurostat")
        yields = pd.DataFrame(index=raw_data.index, columns=["1-year", "5-year", "10-year"])
        yields["1-year"]=raw_data.iloc[:, 0]
        yields["5-year"]=raw_data.iloc[:, 2]
        yields["10-year"]=raw_data.iloc[:, 1]

        yields.loc[date].plot(label=date)
        if save is True:
            mpl.savefig("yield_curve_euro.png", dpi=300)
        if show is True:
            mpl.show()

        return yields

    def ten_year_bond_yields_eu(self):
        yields=pdr.DataReader("teimf050", "eurostat")

        countries=list()
        for i in range(len(yields.columns)):
            countries[i]=yields.columns[i][1]
        yields.columns=countries

        return yields

    def commodities(self):
        tickers=["MCL=F", "NG=F", "ZW=F", "HG=F", "GC=F", "SI=F"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            future=yf.Ticker(ticker)
            current_quotes[ticker]=future.info["regularMarketPrice"]

        names=["Oil Future", "Natural Gas Future", "Wheat Future", "Copper Future", "Gold Spot", "Silver Spot"]
        current_quotes.reindex(names)

        return current_quotes

    def vix(self, show=True, save=False):
        vix_data = yf.download("VIX")
        vix_data["Adj Close"].plot(title="CBOE Volatility Index", xlabel="Date", ylabel="VIX", legen=None)
        if save is True:
            mpl.savefig("vix.png", dpi=300)
        if show is True:
            mpl.show()

    def monetary_us(self):
        monetary_us_codes = ["DFF", "REAINTRATREARAT1YE", "WM1NS", "M2SL", "MABMM301USM189S", "M1V", "M2V"]
        monetary_us = pdr.DataReader(monetary_us_codes, "fred")
        return monetary_us

    def macroeconomic_us(self):
        macreconomic_us_codes = ["GDPC1", "UNRATE", "M318501Q027NBEA", "GFDEBTN", "BOPGSTB", "FPCPITOTLZGUSA", "CORESTICKM159SFRBATL"]
        macroeconomic_us =pdr.DataReader(macreconomic_us_codes, "fred")
        return macroeconomic_us

    def breakeven_inflations_us(self):
        breakeven_inflations_us_codes = ["T5YIE", "T10YIE", "T20YIEM", "T30YIEM"]
        breakeven_inflations_us = pdr.DataReader(breakeven_inflations_us_codes, "fred")
        return breakeven_inflations_us

    def important_rates_us(self):
        rates_codes = ["SOFR", "DPRIME", "MORTGAGE30US"]
        important_rates = pdr.DataReader(rates_codes, "fred")
        return important_rates
        # disount window at FED?
