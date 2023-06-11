import os
from datetime import datetime
import yfinance as yf


class GetData:
    def __init__(
        self,
        tickers,
        start="1970-01-02",
        end=str(datetime.now())[0:10],
        interval="1d",
        **kwargs
    ):
        if len(tickers) == 1:
            self._data = yf.Ticker(tickers[0]).history(
                start=start, end=end, interval=interval, **kwargs
            )
        elif len(tickers) > 1:
            self._data = yf.Tickers(tickers).history(
                start=start, end=end, interval=interval, **kwargs
            )

    @property
    def data(self):
        return self._data

    @property
    def close(self):
        return self._data["Close"]

    def save_all_long(self):
        data_long = (
            self.data.stack(level=1)
            .reset_index(1)
            .rename(columns={"Symbols": "Ticker"})
            .sort_values("Ticker")
        )
        data_long.to_csv("all_tickers_data_long.csv")

    def save_all_wide(self):
        self.data.to_csv("all_tickers_data_wide.csv")

    def save_close_only(self):
        self.close.to_csv("close_only.csv")

    def save_separately(self):
        os.mkdir("tickers_data")
        os.chdir("tickers_data")
        current_data = self.data
        current_data.columns = self.data.columns.swaplevel("Symbols", "Attributes")
        for ticker in current_data.columns:
            time_series = current_data[ticker]
            time_series.to_csv("%s_data.csv" % ticker)
