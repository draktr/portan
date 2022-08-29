import os
import pandas_datareader as pdr
from datetime import datetime


class GetData():
    def __init__(self,
                 tickers,
                 start="1970-01-02",
                 end=str(datetime.now())[0:10],
                 data_source="yahoo"):

        self.data=pdr.DataReader(tickers, start=start, end=end, data_source=data_source)

    def get_dataframe(self):
        return self.data

    def save_all_long(self):
        data_long=self.data.stack(level=1).reset_index(1).rename(columns={"Symbols": "Ticker"}).sort_values("Ticker")
        data_long.to_csv("all_tickers_data_long.csv")

    def save_all_wide(self):
        self.data.to_csv("all_tickers_data_wide.csv")

    def save_adj_close_only(self):
        adj_close_only=self.data["Adj Close"]
        adj_close_only.to_csv("adj_close_only.csv")

    def save_separately(self):
        os.mkdir("tickers_data")
        os.chdir("tickers_data")
        current_data=self.data
        current_data.columns=self.data.columns.swaplevel("Symbols", "Attributes")
        for ticker in current_data.columns:
            time_series=current_data[ticker]
            time_series.to_csv("%s_data.csv"%ticker)
