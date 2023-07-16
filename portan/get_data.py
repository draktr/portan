"""
`get_data.py` module contains `GetData` class
for downloading and exporting financial data
"""


import os
from datetime import datetime
import yfinance as yf
from portan import _checks


CURRENT_DATE = str(datetime.now())[0:10]


class GetData:
    """
    portan.GetData object allows downloading and exporting of financial data with `yfinance`

    - Properties

        - `tickers`/`tickers` - `yfinance.Ticker`/`yfinance.Tickers` object
        - `data` - Full DataFrame of the downloaded data
        - `close` - Assets prices at trading close
    """

    def __init__(
        self, tickers, start="1970-01-02", end=CURRENT_DATE, interval="1d", **kwargs
    ):
        """
        Initialtes GetData object by downloading data from `yfinance` which the user can use within `Python` or save as `.csv`

        :param tickers: Tickers of assets for which data is to be downloaded
        :type tickers: list, np.ndarray, pd.Series, pd.DataFrame
        :param start: Download start date for the data, defaults to "1970-01-02"
        :type start: str (YYYY-MM-DD) or `datetime.datetime()` or `pd.Timestamp`, optional
        :param end: Download end date for the data, defaults to CURRENT_DATE
        :type end: str (YYYY-MM-DD) or `datetime.datetime()` or `pd.Timestamp`, optional
        :param interval: Data interval. Valid intervals are: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo', defaults to "1d"
        :type interval: str, optional
        """

        tickers = _checks._check_get_data(tickers, start, end, interval)

        if len(tickers) == 1:
            self.ticker = yf.Ticker(tickers[0])
            self._data = self.ticker.history(
                start=start, end=end, interval=interval, **kwargs
            )
        elif len(tickers) > 1:
            self.tickers = yf.Tickers(tickers)
            self._data = self.tickers.history(
                start=start, end=end, interval=interval, **kwargs
            ).reindex(columns=tickers, level=1)

    @property
    def data(self):
        """
        Gives access to the full DataFrame of the downloaded data

        :return: Data downloaded
        :rtype: pd.DataFrame
        """

        return self._data

    @property
    def close(self):
        """
        Gives access to the assets prices at trading close

        :return: Data downloaded
        :rtype: pd.DataFrame
        """

        return self._data["Close"]

    def save_long(self):
        """
        Saves downloaded data as `.csv` in long format
        """

        data_long = (
            self.data.stack(level=1)
            .reset_index(1)
            .rename(columns={"Symbols": "Ticker"})
            .sort_values("Ticker")
        )
        data_long.to_csv("all_tickers_data_long.csv")

    def save_wide(self):
        """
        Saves downloaded data as `.csv` in wide format
        """

        self.data.to_csv("all_tickers_data_wide.csv")

    def save_close(self):
        """
        Saves trading close prices as `.csv`
        """

        self.close.to_csv("close_only.csv")

    def save_separately(self):
        """
        Saves trading data as `.csv` for each asset separately
        """

        os.mkdir("tickers_data")
        os.chdir("tickers_data")
        current_data = self.data
        current_data.columns = self.data.columns.swaplevel("Symbols", "Attributes")
        for ticker in current_data.columns:
            time_series = current_data[ticker]
            time_series.to_csv("%s_data.csv" % ticker)
