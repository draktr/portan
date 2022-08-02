import pandas as pd
import matplotlib.pyplot as mpl
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader as pdr

class MarketReporter():
    def __init__(self) -> None:
        self.today=datetime.today()
        self.week_ago=self.today-timedelta(weeks=1)
        self.month_ago=self.today-timedelta(days=30)

    def majors(self):
        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        current_quotes.index=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR"]

        return current_quotes

    def majors_long(self):
        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X", "USDSGD=X", "USDGBP=X" "^RUT", "^FTSE", "^STOXX50E", "^GDAXI", "^HSI", "^STI", "000001.SS", "399001.SZ"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        current_quotes.index=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR", "USDSGD", "USDGBP", "Russell 2000", "FTSE 100", "Euro STOXX 50", "DAX 40", "Hang Seng Index", "The Straits Times Index", "Shanghai Composite", "Shenzhen Index"]

        return current_quotes

    def commodities(self):
        tickers=["MCL=F", "NG=F", "ZW=F", "HG=F", "GC=F", "SI=F"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            future=yf.Ticker(ticker)
            current_quotes[ticker]=future.info["regularMarketPrice"]

        current_quotes.index=["Oil Future", "Natural Gas Future", "Wheat Future", "Copper Future", "Gold Spot", "Silver Spot"]

        return current_quotes

    def vix(self, show=True, save=False):
        vix_data = yf.download("VIX")
        vix_data["Adj Close"].plot(title="CBOE Volatility Index", xlabel="Date", ylabel="VIX", legen=None)
        if save is True:
            mpl.savefig("vix.png", dpi=300)
        if show is True:
            mpl.show()

    def yield_curve_us(self, date=None, show=True, save=False):
        treasuries = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]

        if date is None:
            yields = pdr.DataReader(treasuries, "fred", start=self.week_ago)
            yields.columns = ["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", "5-year", "7-year", "10-year", "20-year", "30-year"]
            yields.iloc[-1].plot(label=yields.index[-1])

        else:
            yields = pdr.DataReader(treasuries, "fred")
            yields.columns = ["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", "5-year", "7-year", "10-year", "20-year", "30-year"]
            yields.loc[date].plot(label=date)

        print(yields)

        if save is True:
            mpl.savefig("yield_curve_us.png", dpi=300)
        if show is True:
            mpl.show()

        return yields

    def yield_curve_euro(self, date=None, show=True, save=False):
        if date is None:
            raw_data = pdr.DataReader("teimf060", "eurostat", start=self.week_ago)
            yields = pd.DataFrame(index=raw_data.index, columns=["1-year", "5-year", "10-year"])
            yields["1-year"]=raw_data.iloc[:, 0]
            yields["5-year"]=raw_data.iloc[:, 2]
            yields["10-year"]=raw_data.iloc[:, 1]
            yields.iloc[-1].plot(label=yields.index[-1])
        else:
            raw_data = pdr.DataReader("teimf060", "eurostat")
            yields = pd.DataFrame(index=raw_data.index, columns=["1-year", "5-year", "10-year"])
            yields["1-year"]=raw_data.iloc[:, 0]
            yields["5-year"]=raw_data.iloc[:, 2]
            yields["10-year"]=raw_data.iloc[:, 1]
            yields.loc[date].plot(label=date)

        print(yields)

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

    def monetary_us(self):
        monetary_us_codes = ["DFF", "REAINTRATREARAT1YE", "WM1NS", "M2SL", "MABMM301USM189S", "M1V", "M2V"]
        monetary_us = pdr.DataReader(monetary_us_codes, "fred")
        return monetary_us

    def macroeconomic_us(self):
        macreconomic_codes = ["GDPC1", "UNRATE", "M318501Q027NBEA", "GFDEBTN", "BOPGSTB", "FPCPITOTLZGUSA", "CORESTICKM159SFRBATL"]
        macroeconomic_us =pdr.DataReader(macreconomic_codes, "fred")
        return macroeconomic_us

    def inflation_expectations_us(self):
        expectations_codes = ["EXPINF5YR", "EXPINF10YR", "EXPINF20YR", "EXPINF30YR" "T5YIE", "T10YIE", "T20YIEM", "T30YIEM"]
        expectations = pdr.DataReader(expectations_codes, "fred")
        return expectations

    def important_rates_us(self):
        rates_codes = ["DFF", "SOFR", "DPRIME", "MORTGAGE30US"]
        rates = pdr.DataReader(rates_codes, "fred")
        return rates

    def euribor(self):
        euribor_codes = ["ECB/RTD_M_S0_N_C_EUR1M_E", "ECB/RTD_M_S0_N_C_EUR3M_E", "ECB/RTD_M_S0_N_C_EUR6M_E", "ECB/RTD_M_S0_N_C_EUR1Y_E"]
        euribor = pdr.DataReader(euribor_codes, "quandl")
        return euribor

    def corporate_yields(self):
        yields_codes = ["AAA", "DBAA", "BAMLH0A0HYM2EY", "BAMLH0A3HYCEY"]
        yields = pdr.DataReader(yields_codes, "fred")
        return yields

    def credit_spreads_us(self):
        spreads_codes = ["AAA10Y", "BAA10Y", "BAMLH0A0HYM2", "BAMLH0A3HYC"]
        spreads = pdr.DataReader(spreads_codes, "fred")
        return spreads
