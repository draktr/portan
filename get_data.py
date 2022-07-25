import os
import pandas as pd
import yfinance as yf

class GetData():
    def __init__(self, ticker, begin, end, period=None):
        self.ticker=ticker
        self.begin=begin
        self.end=end
        self.period=period
        self.df_list=list()

        if self.period==None:
            for ticker in self.ticker:
                time_series = yf.download(ticker, group_by="Ticker", begin=self.begin, end=self.end)
                time_series["ticker"] = ticker
                self.df_list.append(time_series)

        else:
            for ticker in self.ticker:
                time_series = yf.download(ticker, group_by="Ticker", period=self.period)
                time_series["ticker"] = ticker
                self.df_list.append(time_series)

        # combine all dataframes into a single dataframe
        self.data = pd.concat(self.df_list)

    def save_together(self):
        self.data.to_csv("all_tickers_data.csv")

    def save_separately(self):
        os.mkdir("tickers_data")
        os.chdir("tickers_data")
        for i in range(len(self.ticker)):
            data = self.df_list[i]
            ticker = data["ticker"][0]
            data.to_csv("%s_data.csv"%ticker)
