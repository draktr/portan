import pandas as pd
import yfinance as yf

class GetData():
    def _init_(self, ticker, begin, end, period):
        self.ticker=ticker
        self.begin=begin
        self.end=end
        self.period=period

    def get_time_series(self):
        if self.period==None:
            df_list=list()
            for ticker in self.ticker:
                time_series = yf.download(ticker, group_by="Ticker", begin=self.begin, end=self.end)
                time_series["ticker"] = ticker
                df_list.append(time_series)

        else:
            df_list=list()
            for ticker in self.ticker:
                time_series = yf.download(ticker, group_by="Ticker", period=self.period)
                time_series["ticker"] = ticker
                df_list.append(time_series)

        # combine all dataframes into a single dataframe
        self.data = pd.concat(df_list)
        return self.data

    def save_to_csv(self):
        # save to csv
        self.data.to_csv("ticker.csv")
