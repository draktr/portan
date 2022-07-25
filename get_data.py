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
                time_series = yf.download(ticker, begin=self.begin, end=self.end)
                time_series["ticker"] = ticker
                self.df_list.append(time_series)
        else:
            for ticker in self.ticker:
                time_series = yf.download(ticker, period=self.period)
                time_series["ticker"] = ticker
                self.df_list.append(time_series)

    def save_all_long(self):
        data_long = pd.concat(self.df_list, axis=0)
        data_long.to_csv("all_tickers_data_long.csv")

    def save_all_wide(self):
        data_wide = pd.concat(self.df_list, axis=1)
        data_wide.to_csv("all_tickers_data_wide.csv")

    def save_adj_close_only(self):
        adj_close_only = pd.DataFrame(columns=self.ticker)
        for ticker, i in zip(self.ticker, range(len(self.df_list))):
            adj_close_only[ticker] = self.df_list[i]["Adj Close"]
        adj_close_only.to_csv("adj_close_only.csv")

    def save_separately(self):
        os.mkdir("tickers_data")
        os.chdir("tickers_data")
        for i in range(len(self.ticker)):
            data = self.df_list[i]
            ticker = data["ticker"][0]
            data.to_csv("%s_data.csv"%ticker)
