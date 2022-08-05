import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader as pdr

class MarketReporter():

    def __init__(self) -> None:
        self.today=datetime.today()
        self.week_ago=self.today-timedelta(weeks=1)
        self.month_ago=self.today-timedelta(days=30)

    def majors(self):
        """
        Returns current market prices of the most important major quotes
        (S&P500, DJI, Nasdaq Composite, 10-year US Treasury Yield, VIX, USDEUR)

        Returns:
            pd.Series: Current market prices
        """

        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        current_quotes.index=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", \
                              "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR"]

        return current_quotes

    def majors_long(self):
        """
        Returns current market prices of major quotes

        Returns:
            pd.Series: Current market prices
        """

        tickers=["^GSPC", "^DJI", "^IXIC", "^TNX", "VIX", "USDEUR=X", "USDSGD=X", "USDGBP=X", \
                 "^RUT", "^FTSE", "^STOXX50E", "^GDAXI", "^HSI", "^STI", "000001.SS", "399001.SZ"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            security=yf.Ticker(ticker)
            current_quotes[ticker]=security.info["regularMarketPrice"]

        current_quotes.index=["S&P500", "Dow Jones Industrial Average", "NASDAQ Composite", \
                              "10-year US Treasury Yield", "CBOE Volatility Index", "USDEUR", \
                              "USDSGD", "USDGBP", "Russell 2000", "FTSE 100", "Euro STOXX 50", \
                              "DAX 40", "Hang Seng Index", "The Straits Times Index", \
                              "Shanghai Composite", "Shenzhen Index"]

        return current_quotes

    def commodities(self):
        """
        Returns curent market prices of futures contracts for major commodities
        (Oil, Natural Gas, Wheat, Copper, Gold, Silver)

        Returns:
            pd.Series: Current market prices
        """

        tickers=["MCL=F", "NG=F", "ZW=F", "HG=F", "GC=F", "SI=F"]
        current_quotes = pd.Series(index=tickers)

        for ticker in tickers:
            future=yf.Ticker(ticker)
            current_quotes[ticker]=future.info["regularMarketPrice"]

        current_quotes.index=["Oil Future", "Natural Gas Future", "Wheat Future", "Copper Future", \
                              "Gold Spot", "Silver Spot"]

        return current_quotes

    def quotes_custom(self, tickers, info="regularMarketPrice", long_names=False):
        """
        Returns current market prices of a custom list of securities. Only works for shares.
        For other securities it returns empty dataframe.

        Args:
            tickers (list): List of tickers
            info (str, optional): Information about stock requested. Defaults to "regularMarketPrice".
            long_names (bool, optional): Whether to index the series with a long name or ticker. Defaults to False.

        Returns:
            pd.DataFrame: Current Market Prices
        """

        quotes=pdr.get_quote_yahoo(tickers)
        if long_names is True:
            quotes.index=quotes["longName"]

        return quotes[info]

    def vix(self, show=True, save=False):
        """
        Plots Adjusted Close price of CBOE Volatility Index

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot on storage. Defaults to False.
        """

        vix_data = yf.download("VIX")
        vix_data["Adj Close"].plot(title="CBOE Volatility Index", xlabel="Date", ylabel="VIX", legen=None)
        if save is True:
            plt.savefig("vix.png", dpi=300)
        if show is True:
            plt.show()

    def yield_curve_us(self, date=None, show=True, save=False):
        """
        Plots the yield curve for the US Treasuries

        Args:
            date (_type_, optional): Date of the yields. Defaults to None (last trading day).
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot on storage. Defaults to False.


        Returns:
            pd.DataFrame: Bond yields
        """

        treasuries = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", \
                      "DGS10", "DGS20", "DGS30"]

        if date is None:
            yields = pdr.DataReader(treasuries, "fred", start=self.week_ago)
            yields.columns = ["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", \
                              "5-year", "7-year", "10-year", "20-year", "30-year"]
            yields.iloc[-1].plot(label=yields.index[-1])

        else:
            yields = pdr.DataReader(treasuries, "fred")
            yields.columns = ["1-month", "3-month", "6-month", "1-year", "2-year", "3-year", \
                              "5-year", "7-year", "10-year", "20-year", "30-year"]
            yields.loc[date].plot(label=date)

        print(yields)

        if save is True:
            plt.savefig("yield_curve_us.png", dpi=300)
        if show is True:
            plt.show()

        return yields

    def yield_curve_euro(self, date=None, show=True, save=False):
        """
        Plots the yield curve for the Euro Area

        Args:
            date (_type_, optional): Date of the yields. Defaults to None (last trading day).
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot on storage. Defaults to False.


        Returns:
            pd.DataFrame: Bond yields
        """

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
            plt.savefig("yield_curve_euro.png", dpi=300)
        if show is True:
            plt.show()

        return yields

    def ten_year_bond_yields_eu(self):
        """
        Returns 10-year bond yields for all EU countries.

        Returns:
            pd.DataFrame: Bond yields
        """

        yields=pdr.DataReader("teimf050", "eurostat")

        countries=list()
        for i in range(len(yields.columns)):
            countries[i]=yields.columns[i][1]
        yields.columns=countries

        return yields

    def monetary_us(self):
        """
        Returns important monetary statistics for the US.

        Returns:
            pd.DataFrame: Monetary statistics
        """

        monetary_us_codes = ["DFF", "REAINTRATREARAT1YE", "WM1NS", "M2SL", "MABMM301USM189S", "M1V", "M2V"]
        monetary_us = pdr.DataReader(monetary_us_codes, "fred")
        return monetary_us

    def macroeconomic_us(self):
        """
        Returns important macroeconomic statistics for the US.

        Returns:
            pd.DataFrame: Macroeconomic statistics
        """

        macreconomic_codes = ["GDPC1", "UNRATE", "M318501Q027NBEA", "GFDEBTN", "BOPGSTB", "FPCPITOTLZGUSA", \
                              "CORESTICKM159SFRBATL"]
        macroeconomic_us =pdr.DataReader(macreconomic_codes, "fred")
        return macroeconomic_us

    def inflation_expectations_us(self):
        """
        Returns different measures of inflation rate expectations for the US.

        Returns:
            pd.DataFrame: Inflation rate expectations
        """

        expectations_codes = ["EXPINF5YR", "EXPINF10YR", "EXPINF20YR", "EXPINF30YR" "T5YIE", "T10YIE", \
                              "T20YIEM", "T30YIEM"]
        expectations = pdr.DataReader(expectations_codes, "fred")
        return expectations

    def important_rates_us(self):
        """
        Returns important interest rates in the US.

        Returns:
            pd.DataFrame: Interest rates
        """

        rates_codes = ["DFF", "SOFR", "DPRIME", "MORTGAGE30US"]
        rates = pdr.DataReader(rates_codes, "fred")
        return rates

    def euribor(self):
        """
        Returns 1-month, 3-months, 6-months, 1-year Euribor rates.

        Returns:
            pd.DataFrame: Euribor rates
        """

        euribor_codes = ["ECB/RTD_M_S0_N_C_EUR1M_E", "ECB/RTD_M_S0_N_C_EUR3M_E", "ECB/RTD_M_S0_N_C_EUR6M_E", \
                         "ECB/RTD_M_S0_N_C_EUR1Y_E"]
        euribor = pdr.DataReader(euribor_codes, "quandl")
        return euribor

    def corporate_yields(self):
        """
        Returns important corporate bond yields in the US.

        Returns:
            pd.DataFrame: Bond yields
        """
        yields_codes = ["AAA", "DBAA", "BAMLH0A0HYM2EY", "BAMLH0A3HYCEY"]
        yields = pdr.DataReader(yields_codes, "fred")
        return yields

    def credit_spreads_us(self):
        """
        Returns important credit spreads in the us.

        Returns:
            pd.DataFrame: Spreads
        """
        spreads_codes = ["AAA10Y", "BAA10Y", "BAMLH0A0HYM2", "BAMLH0A3HYC"]
        spreads = pdr.DataReader(spreads_codes, "fred")
        return spreads
