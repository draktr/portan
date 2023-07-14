"""
`analytics.py` module contains `portan.Analytics` class
for analyzing portfolio performance
"""


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn import covariance
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.tsa import stattools
from itertools import repeat
from datetime import datetime
import warnings
from portan import _checks


STYLE = "./portan/portan_style.mplstyle"
CURRENT_DATE = str(datetime.now())[0:10]


class Analytics:
    """
    `portan.Analytics object constructs framework for portfolio analytics

    - Properties

        - `prices` - Assets prices
        - `assets_returns` - Assets returns
        - `tickers` - Assets tickers
        - `weights` - Assets weights
        - `assets_info` - Information about assets
        - `assets_names` - Assets names
        - `name` - Name of the portfolio
        - `initial_aum` - Initial Assets Under Management (AUM)
        - `frequency` - Data frequency, number of data observations in a year. Used for annualization.
        - `allocation_funds` - Allocation of funds into each asset
        - `allocation_assets` - Number of each asset in the portfolio
        - `state` - State of the each asset and whole portfolio time-series
        - `returns` - Portfolio returns time-series
        - `cumulative_returns` - Cumulative portfolio returns time-series
        - `mean` - Mean portfolio return
        - `arithmetic_mean` - Annualized arithmetic (not compounded) mean return
        - `geometric_mean` - Annualized geometric (compounded) mean return
        - `volatility` - Portfolio volatility (standard deviation)
        - `annual_volatility` - Annualized portfolio volatility (standard deviation)
        - `skewness` - Returns distribution skewness
        - `kurtosis` - Returns distribution kurtosis
        - `min_aum` - Minimum AUM during the data period
        - `max_aum` - Maximum AUM during the data period
        - `mean_aum` - Average AUM during the data period
        - `final_aum` - AUM at the last point of the data period
        - `benchmark_prices` - Benchmark assets prices
        - `benchmark_weights` - Benchmark assets weights
        - `benchmark_name` - Benchmark name
        - `benchmark_assets_returns` - Benchmark assets returns
        - `benchmark_returns` - Benchmark returns time-series
        - `benchmark_mean` - Mean benchmark return
        - `benchmark_arithmetic_mean` - Annualized arithmetic (not compounded) mean benchmark return
        - `benchmark_geometric_mean` - Annualized geometric (compounded) mean benchmark return
    """

    def __init__(
        self,
        tickers=None,
        prices=None,
        weights=None,
        benchmark_tickers=None,
        benchmark_prices=None,
        benchmark_weights=None,
        name="Investment Portfolio",
        benchmark_name="Benchmark Portfolio",
        initial_aum=10000,
        frequency=252,
        start="1970-01-02",
        end=CURRENT_DATE,
        interval="1d",
    ) -> None:
        """
        Initiates `portan.Analytics` object

        :param tickers: Assets tickers. Used to download assets prices time-series with `yfinance` if `prices=None`, defaults to None
        :type tickers: _type_, optional
        :param prices: Assets prices time-series. Optional, as assets prices can be downloaded by providing `tickers` argument. Especially useful when assets prices aren't available on `yfinance` or assets aren't public, defaults to None
        :type prices: _type_, optional
        :param weights: Assets weights. Mandatory argument. Default set to `None` only because of the ordering of arguments, defaults to None
        :type weights: _type_, optional
        :param benchmark_tickers: Benchmark assets tickers. Used to download benchmark assets prices time-series with `yfinance` if `benchmark_prices=None`, defaults to None
        :type benchmark_tickers: _type_, optional
        :param benchmark_prices: Benchmark assets prices time-series. Optional, as benchmark assets prices can be downloaded by providing `benchmark_tickers` argument. Especially useful when benchmark assets prices aren't available on `yfinance` or benchmark assets aren't public, defaults to None
        :type benchmark_prices: _type_, optional
        :param benchmark_weights: Benchmark assets weights. Mandatory argument. Default set to `None` only because of the ordering of arguments, defaults to None
        :type benchmark_weights: _type_, optional
        :param name: Portfolio name, defaults to "Investment Portfolio"
        :type name: str, optional
        :param benchmark_name: Benchmark name, defaults to "Benchmark Portfolio"
        :type benchmark_name: str, optional
        :param initial_aum: Initial Assets Under Management (AUM), defaults to 10000
        :type initial_aum: int, optional
        :param frequency: Data frequency, number of data observations in a year, defaults to 252
        :type frequency: int, optional
        :param start: Start date used for downloading assets prices data if `tickers` and/or `benchmark_tickers` arguments are provided, and `prices` and/or `benchmark_prices` arguments `None`, defaults to "1970-01-02"
        :type start: str (YYYY-MM-DD) or `datetime.datetime()` or `pd.Timestamp`, optional
        :param end: End date used for downloading assets prices data if `tickers` and/or `benchmark_tickers` arguments are provided, and `prices` and/or `benchmark_prices` arguments `None`, defaults to CURRENT_DATE
        :type end: str (YYYY-MM-DD) or `datetime.datetime()` or `pd.Timestamp`, optional
        :param interval: Data interval used for downloading assets prices data if `tickers` and/or `benchmark_tickers` arguments are provided, and `prices` and/or `benchmark_prices` arguments `None`. Valid intervals are: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo', defaults to "1d"
        :type interval: str, optional
        """

        prices, weights, benchmark_prices, benchmark_weights = _checks._check_init(
            tickers,
            prices,
            weights,
            benchmark_tickers,
            benchmark_prices,
            benchmark_weights,
            name,
            benchmark_name,
            initial_aum,
            frequency,
            start,
            end,
            interval,
        )

        self.prices = prices
        self.assets_returns = self.prices.pct_change().drop(self.prices.index[0])
        self.tickers = self.prices.columns.tolist()
        self.weights = pd.Series(weights, index=self.tickers)
        self.assets_info = np.empty(len(self.tickers), dtype=object)
        self.assets_names = np.empty(len(self.tickers), dtype="<U64")
        try:
            for i, ticker in enumerate(self.tickers):
                self.assets_info[i] = yf.Ticker(ticker).info
                self.assets_names[i] = self.assets_info[i]["longName"]
            self.assets_names = self.assets_names.tolist()
        except Exception:
            warnings.warn(
                "Couldn't obtain `assets_info` and `assets_names` from `yfinance`. Use `_set_info_names()` to set `assets_info` and `assets_names` properties"
            )
        self.name = name
        self.initial_aum = initial_aum
        self.frequency = frequency

        self.allocation_funds = pd.Series(
            np.multiply(self.initial_aum, self.weights), index=self.tickers
        )

        self.allocation_assets = pd.Series(
            np.divide(self.allocation_funds, self.prices.iloc[0].T), index=self.tickers
        )

        self.state = pd.DataFrame(
            np.multiply(self.prices, self.allocation_assets),
            index=self.prices.index,
            columns=self.tickers,
        )
        self.state[self.name] = self.state.sum(axis=1)

        if len(weights) == 1:
            self.returns = pd.DataFrame(
                np.dot(self.assets_returns.to_numpy(), self.weights[0]),
                index=self.assets_returns.index,
                columns=[self.name],
            )
        elif len(weights) > 1:
            self.returns = pd.DataFrame(
                np.dot(self.assets_returns.to_numpy(), self.weights),
                index=self.assets_returns.index,
                columns=[self.name],
            )

        self.cumulative_returns = (self.returns + 1).cumprod()

        self.mean = self.returns.mean()[0]
        self.arithmetic_mean = self.mean * self.frequency
        self.geometric_mean = (
            (1 + self.returns).prod() ** (self.frequency / self.returns.shape[0]) - 1
        )[0]

        self.volatility = self.returns.std()[0]
        self.annual_volatility = self.volatility * np.sqrt(self.frequency)

        self.skewness = stats.skew(self.returns)[0]
        self.kurtosis = stats.kurtosis(self.returns)[0]

        self.min_aum = self.state[self.name].min()
        self.max_aum = self.state[self.name].max()
        self.mean_aum = self.state[self.name].mean()
        self.final_aum = self.state.iloc[-1, -1]

        if benchmark_prices is None and benchmark_weights is None:
            self.benchmark_prices = benchmark_prices
            self.benchmark_weights = benchmark_weights
            self.benchmark_name = benchmark_name

            self.benchmark_assets_returns = None
            self.benchmark_returns = None

            self.benchmark_mean = None
            self.benchmark_arithmetic_mean = None
            self.benchmark_geometric_mean = None
        elif benchmark_prices is not None and benchmark_weights is not None:
            self.benchmark_prices = benchmark_prices
            self.benchmark_weights = pd.Series(
                benchmark_weights, index=self.benchmark_prices.columns
            )
            self.benchmark_name = benchmark_name

            self.benchmark_assets_returns = self.benchmark_prices.pct_change().drop(
                self.benchmark_prices.index[0]
            )

            if len(self.benchmark_weights) == 1:
                self.benchmark_returns = pd.DataFrame(
                    np.dot(
                        self.benchmark_assets_returns.to_numpy(),
                        self.benchmark_weights[0],
                    ),
                    index=self.benchmark_assets_returns.index,
                    columns=[self.benchmark_name],
                )
            elif len(self.benchmark_weights) > 1:
                self.benchmark_returns = pd.DataFrame(
                    np.dot(
                        self.benchmark_assets_returns.to_numpy(),
                        self.benchmark_weights,
                    ),
                    index=self.benchmark_assets_returns.index,
                    columns=[self.benchmark_name],
                )

            self.benchmark_mean = self.benchmark_returns.mean()[0]
            self.benchmark_arithmetic_mean = self.benchmark_mean * self.frequency
            self.benchmark_geometric_mean = (
                (1 + self.benchmark_returns).prod()
                ** (self.frequency / self.benchmark_returns.shape[0])
                - 1
            )[0]

    def _set_benchmark(
        self,
        benchmark_tickers=None,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
        start="1970-01-02",
        end=CURRENT_DATE,
        interval="1d",
    ):
        benchmark_prices, benchmark_weights, prices = _checks._check_benchmark(
            benchmark_tickers,
            benchmark_prices,
            benchmark_weights,
            benchmark_name,
            self.prices,
            start,
            end,
            interval,
        )

        self.benchmark_prices = benchmark_prices
        self.benchmark_weights = pd.Series(
            benchmark_weights, index=self.benchmark_prices.columns
        )
        self.benchmark_name = benchmark_name
        self.prices = prices

        self.benchmark_assets_returns = self.benchmark_prices.pct_change().drop(
            self.benchmark_prices.index[0]
        )

        if len(self.benchmark_weights) == 1:
            self.benchmark_returns = pd.DataFrame(
                np.dot(
                    self.benchmark_assets_returns.to_numpy(),
                    self.benchmark_weights[0],
                ),
                index=self.benchmark_assets_returns.index,
                columns=[self.benchmark_name],
            )
        elif len(self.benchmark_weights) > 1:
            self.benchmark_returns = pd.DataFrame(
                np.dot(
                    self.benchmark_assets_returns.to_numpy(),
                    self.benchmark_weights,
                ),
                index=self.benchmark_assets_returns.index,
                columns=[self.benchmark_name],
            )

        self.benchmark_mean = self.benchmark_returns.mean()[0]
        self.benchmark_arithmetic_mean = self.benchmark_mean * self.frequency
        self.benchmark_geometric_mean = (
            (1 + self.benchmark_returns).prod()
            ** (self.frequency / self.benchmark_returns.shape[0])
            - 1
        )[0]

    def _set_info_names(self, assets_info=None, assets_names=None):
        print("(Re)setting `assets_info` and `assets_names` properties")
        self.assets_info = assets_info
        self.assets_names = assets_names

    def _rate_conversion(self, annual_rate):
        return (annual_rate + 1) ** (1 / self.frequency) - 1

    def excess_mar(self, annual_mar=0.03, annual=True, compounding=True):
        """
        Calculates excess mean return above Minimum Accepted Return (MAR)

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Excess mean return above MAR
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        if annual and compounding:
            excess_return = self.geometric_mean - annual_mar
        if annual and not compounding:
            excess_return = self.arithmetic_mean - annual_mar
        if not annual:
            mar = self._rate_conversion(annual_mar)
            excess_return = self.mean - mar

        return excess_return

    def net_return(self, percentage=False):
        """
        Calculates net investment return

        :param percentage: Whether to calculate in percentage or absolute terms, defaults to False
        :type percentage: bool, optional
        :return: Net investment return
        :rtype: float
        """

        _checks._check_booleans(percentage=percentage)

        final_aum = self.final_aum

        if not percentage:
            net_return = final_aum - self.initial_aum
        else:
            net_return = (final_aum - self.initial_aum) / self.initial_aum

        return net_return

    def distribution_test(self, test="dagostino-pearson", distribution="norm"):
        """
        Tests for the probability distribution of the investment returns

        :param test: Statistical test to be used. Available are `"dagostino-pearson"`,
                     `"kolomogorov-smirnov"`, `"lilliefors"`, `"shapiro-wilk"`,
                     `"jarque-barre"`, `"anderson-darling"`,
                     defaults to "dagostino-pearson"
        :type test: str, optional
        :param distribution: Probability distribution hypothesized,
                             defaults to "norm"
        :type distribution: str, optional
        :raises ValueError: If statistical test is unavailable
        :return: Result of the tests (e.g. test statistic value, p-value, etc.)
        :rtype: TestResult object or tuple
        """

        if test == "dagostino-pearson":
            result = stats.normaltest(self.returns)
        elif test == "kolomogorov-smirnov":
            result = stats.kstest(self.returns, distribution)
        elif test == "lilliefors":
            result = lilliefors(self.returns)
        elif test == "shapiro-wilk":
            result = stats.shapiro(self.returns)
        elif test == "jarque-barre":
            result = stats.jarque_bera(self.returns)
        elif test == "anderson-darling":
            result = stats.anderson(self.returns, distribution)
        else:
            raise ValueError("Statistical test is unavailable.")

        return result

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        annual=True,
        compounding=True,
        **ewm_kwargs,
    ):
        """
        Calculates exponentially weighted mean return

        :param com: Specify decay in terms of center of mass
                    .. math::
                        \alpha = \frac{1}{1+com}
                    for com >= 0, defaults to None
        :type com: float, optional
        :param span: Specify decay in terms of span
                     .. math::
                        \alpha = \frac{2}{span+1}
                     for span >= 1, defaults to None
        :type span: float, optional
        :param halflife: Specify decay in terms of halflife
                         .. math::
                            \alpha = 1-\exp(\frac{-\ln(2)}{halflife}
                         for halflife > 0, defaults to None
        :type halflife: float, optional
        :param alpha: Specify smoothing factor directly, defaults to None
        :type alpha: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Exponentially weighted mean return
        :rtype: float
        """

        _checks._check_rate_arguments(annual=annual, compounding=compounding)

        if annual and compounding:
            mean = (
                1
                + self.returns.ewm(com, span, halflife, alpha, **ewm_kwargs)
                .mean()
                .iloc[-1]
            ) ** self.frequency - 1
        elif annual and not compounding:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **ewm_kwargs)
                .mean()
                .iloc[-1]
                * self.frequency
            )
        elif not annual:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **ewm_kwargs)
                .mean()
                .iloc[-1]
            )

        return mean[0]

    def plot_aum(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots assets under management (AUM) over time

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.state[self.name].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("AUM")
        ax.set_title("Assets Under Management")
        if save:
            plt.savefig(f"{self.name}_aum.png", dpi=300)
        if show:
            plt.show()

    def plot_returns(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio returns over time

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.set_title("Portfolio Returns")
        if save:
            plt.savefig(f"{self.name}_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_returns_distribution(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio returns distribution histogram

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.returns.plot.hist(ax=ax, bins=100)
        ax.set_xlabel("Return")
        ax.set_ylabel("Return Frequency")
        ax.set_title("Portfolio Return Distribution")
        if save:
            plt.savefig(f"{self.name}_return_distribution.png", dpi=300)
        if show:
            plt.show()

    def plot_cumulative_returns(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio cumulative returns over time

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Portfolio Cumulative Returns")
        if save:
            plt.savefig(f"{self.name}_cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_piechart(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio assets pie chart

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        wp = {"linewidth": 1, "edgecolor": "black"}
        explode = tuple(repeat(0.05, len(self.tickers)))

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        pie = ax.pie(
            self.allocation_funds,
            autopct=lambda pct: self._ap(pct, self.allocation_funds),
            explode=explode,
            labels=self.tickers,
            shadow=True,
            startangle=90,
            wedgeprops=wp,
        )
        ax.legend(
            pie[0],
            self.assets_names,
            title="Portfolio Assets",
            loc="upper right",
            bbox_to_anchor=(0.7, 0, 0.5, 1),
        )
        plt.setp(pie[2], size=9, weight="bold")
        ax.set_title(f"{self.name} Asset Distribution")
        if save:
            plt.savefig(f"{self.name}_pie_chart.png", dpi=300)
        if show:
            plt.show()

    def plot_assets_cumulative_returns(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio individual assets cumulative returns over time

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        assets_cumulative_returns = (self.assets_returns + 1).cumprod()

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        assets_cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Assets Cumulative Returns")
        ax.legend(labels=self.assets_names)
        if save:
            plt.savefig(f"{self.name}_assets_cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def _ap(self, pct, all_values):
        absolute = int(pct / 100.0 * np.sum(all_values))

        return "{:.1f}%\n(${:d})".format(pct, absolute)

    def capm(
        self,
        annual_rfr=0.03,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Estimates Capital Asset Pricing Model (CAPM) parameters

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: CAPM alpha, beta, epsilon, R-squared
        :rtype: float, float, np.ndarray, float
        """

        _checks._check_rate_arguments(annual_rfr=annual_rfr)

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        rfr = self._rate_conversion(annual_rfr)
        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(excess_benchmark_returns, excess_returns)
        alpha = model.intercept_[0]
        beta = model.coef_[0][0]
        epsilon = excess_returns.to_numpy() - alpha - beta * excess_benchmark_returns
        r_squared = model.score(excess_benchmark_returns, excess_returns)

        return alpha, beta, epsilon, r_squared

    def plot_capm(
        self,
        annual_rfr=0.03,
        style=STYLE,
        rcParams_update={},
        show=True,
        save=False,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
        **fig_kw,
    ):
        """
        Plots Capital Asset Pricing Model (CAPM) model elements

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        capm = self.capm(annual_rfr, benchmark)

        rfr = self._rate_conversion(annual_rfr)
        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ax.scatter(excess_benchmark_returns, excess_returns, color="b")
        ax.plot(
            excess_benchmark_returns,
            capm[0] + capm[1] * excess_benchmark_returns,
            color="r",
        )
        empty_patch = mpatches.Patch(color="none", visible=False)
        ax.legend(
            handles=[empty_patch, empty_patch],
            labels=[
                r"$\alpha$" + " = " + str(np.round(capm[0], 5)),
                r"$\beta$" + " = " + str(np.round(capm[1], 5)),
            ],
            handlelength=-1,
        )
        ax.set_xlabel("Benchmark Excess Return")
        ax.set_ylabel("Portfolio Excess Return")
        ax.set_title("Portfolio Excess Returns Against Benchmark (CAPM)")
        if save:
            plt.savefig(f"{self.name}_capm.png", dpi=300)
        if show:
            plt.show()

    def sharpe(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        adjusted=False,
        probabilistic=False,
        sharpe_benchmark=0.0,
    ):
        """
        Calculates Sharpe ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param adjusted: Whether to calculate adjusted Sharpe ratio, defaults to False
        :type adjusted: bool, optional
        :param probabilistic: Whether to calculate probabilistic Sharpe ratio, defaults to False
        :type probabilistic: bool, optional
        :param sharpe_benchmark: Benchmark Sharpe ratio for probabilistic Sharpe ratio (if used), defaults to 0.0
        :type sharpe_benchmark: float, optional
        :return: Sharpe ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_sharpe(adjusted=adjusted, probabilistic=probabilistic)

        if annual and compounding:
            sharpe_ratio = (self.geometric_mean - annual_rfr) / self.annual_volatility
        elif annual and not compounding:
            sharpe_ratio = (self.arithmetic_mean - annual_rfr) / self.annual_volatility
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            sharpe_ratio = (self.mean - rfr) / self.volatility

        if adjusted:
            sharpe_ratio = sharpe_ratio * (
                1
                + (self.skewness / 6) * sharpe_ratio
                - ((self.kurtosis - 3) / 24) * sharpe_ratio**2
            )

        if probabilistic:
            sharpe_std = np.sqrt(
                (
                    1
                    + (0.5 * sharpe_ratio**2)
                    - (self.skewness * sharpe_ratio)
                    + (((self.kurtosis - 3) / 4) * sharpe_ratio**2)
                )
                / (self.returns.shape[0] - 1)
            )
            sharpe_ratio = stats.norm.cdf(
                (sharpe_ratio - sharpe_benchmark) / sharpe_std
            )

        return sharpe_ratio

    def capm_return(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Expected excess portfolio return estimated by Capital Asset Pricing Model (CAPM)

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Expected excess portfolio return estimated by CAPM
        :rtype: float
        """

        capm = self.capm(annual_rfr, benchmark)

        if annual and compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_geometric_mean - annual_rfr)
        elif annual and not compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_arithmetic_mean - annual_rfr)
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            mean = rfr + capm[1] * (self.benchmark_mean - rfr)

        return mean

    def excess_benchmark(
        self,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates excess mean return above benchmark mean return

        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Excess mean return above benchmark
        :rtype: float
        """

        _checks._check_rate_arguments(annual=annual, compounding=compounding)
        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        if annual and compounding:
            excess_return = self.geometric_mean - self.benchmark_geometric_mean
        elif annual and not compounding:
            excess_return = self.arithmetic_mean - self.benchmark_arithmetic_mean
        elif not annual:
            excess_return = self.mean - self.benchmark_mean

        return excess_return

    def tracking_error(
        self,
        annual=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates tracking error

        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Tracking error
        :rtype: float
        """

        _checks._check_rate_arguments(annual=annual)
        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        tracking_error = np.std(self.returns - self.benchmark_returns.to_numpy())

        if annual:
            return tracking_error[0] * np.sqrt(self.frequency)
        else:
            return tracking_error[0]

    def information_ratio(
        self,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates information ratio

        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Information ratio
        :rtype: float
        """

        excess_return = self.excess_benchmark(annual, compounding, benchmark)
        tracking_error = self.tracking_error(annual, benchmark)

        information_ratio = excess_return / tracking_error

        return information_ratio

    def volatility_skewness(self, annual_mar=0.03):
        """
        Calculates volatility skewness

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Volatility skewness
        :rtype: float
        """

        upside = self.hpm(annual_mar=annual_mar, moment=2)
        downside = self.lpm(annual_mar=annual_mar, moment=2)
        skewness = upside / downside

        return skewness

    def omega_excess_return(
        self,
        annual_mar=0.03,
        annual=True,
        compounding=True,
    ):
        """
        Calculates omega excess return

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Omega excess return
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_mar=annual_mar, annual=annual, compounding=compounding
        )

        mar = self._rate_conversion(annual_mar)
        days = self.benchmark_returns.shape[0]

        portfolio_downside_risk = self.downside_risk(annual_mar)
        benchmark_downside_risk = np.sqrt(
            (1 / days)
            * np.sum(np.power(np.maximum(mar - self.benchmark_returns, 0), 2))
        )

        if annual and compounding:
            omega_excess_return = (
                self.geometric_mean
                - 3
                * portfolio_downside_risk
                * np.sqrt(self.frequency)
                * benchmark_downside_risk
                * np.sqrt(self.frequency)
            )
        elif annual and not compounding:
            omega_excess_return = (
                self.arithmetic_mean
                - 3
                * portfolio_downside_risk
                * np.sqrt(self.frequency)
                * benchmark_downside_risk
                * np.sqrt(self.frequency)
            )
        elif not annual:
            omega_excess_return = (
                self.mean - 3 * portfolio_downside_risk * benchmark_downside_risk
            )

        return omega_excess_return[0]

    def sortino(self, annual_mar=0.03, annual_rfr=0.03, annual=True, compounding=True):
        """
        Calculates Sortino ratio

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Sortino ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_mar=annual_mar,
            annual_rfr=annual_rfr,
            annual=annual,
            compounding=compounding,
        )

        downside_risk = self.downside_risk(annual_mar)

        if annual and compounding:
            sortino_ratio = (self.geometric_mean - annual_rfr) / downside_risk
        elif annual and not compounding:
            sortino_ratio = (self.arithmetic_mean - annual_rfr) / downside_risk
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            sortino_ratio = (self.mean - rfr) / downside_risk

        return sortino_ratio

    def jensen_alpha(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Jensen alpha

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Jensen alpha
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        capm = self.capm(annual_rfr, benchmark)

        if annual and compounding:
            jensen_alpha = (
                self.geometric_mean
                - annual_rfr
                - capm[1] * (self.benchmark_geometric_mean - annual_rfr)
            )
        elif annual and not compounding:
            jensen_alpha = (
                self.arithmetic_mean
                - annual_rfr
                - capm[1] * (self.benchmark_arithmetic_mean - annual_rfr)
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            jensen_alpha = self.mean - rfr - capm[1] * (self.benchmark_mean - rfr)

        return jensen_alpha

    def treynor(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Treynor ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Treynor ratio
        :rtype: Treynor ratio
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        capm = self.capm(annual_rfr, benchmark)

        if annual and compounding:
            treynor_ratio = (self.geometric_mean - annual_rfr) / capm[1]
        elif annual and not compounding:
            treynor_ratio = (self.arithmetic_mean - annual_rfr) / capm[1]
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            treynor_ratio = (self.mean - rfr) / capm[1]

        return treynor_ratio

    def hpm(self, annual_mar=0.03, moment=3):
        """
        Calculates Higher Partial Moment

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param moment: Moment for calculation, defaults to 3
        :type moment: int, optional
        :return: Higher Partial Moment
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.maximum(self.returns - mar, 0), moment)
        )

        return higher_partial_moment[0]

    def lpm(self, annual_mar=0.03, moment=3):
        """
        Calculates Lower Partial Moment

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param moment: Moment for calculation, defaults to 3
        :type moment: int, optional
        :return: Lower Partial Moment
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.maximum(mar - self.returns, 0), moment)
        )

        return lower_partial_moment[0]

    def kappa(self, annual_mar=0.03, moment=3, annual=True, compounding=True):
        """
        Calculates Kappa

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :param moment: Moment for calculation, defaults to 3
        :type moment: int, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Kappa
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_mar=annual_mar, annual=annual, compounding=compounding
        )
        _checks._check_posints(moment=moment)

        lower_partial_moment = self.lpm(annual_mar, moment)

        if annual and compounding:
            kappa_ratio = (
                100
                * (self.geometric_mean - annual_mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif annual and not compounding:
            kappa_ratio = (
                100
                * (self.arithmetic_mean - annual_mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif not annual:
            mar = self._rate_conversion(annual_mar)
            kappa_ratio = (self.mean - mar) / np.power(
                lower_partial_moment, (1 / moment)
            )

        return kappa_ratio

    def gain_loss(self):
        """
        Calculates Gain-loss ratio

        :return: Gain-loss ratio
        :rtype: float
        """

        higher_partial_moment = self.hpm(annual_mar=0, moment=1)
        lower_partial_moment = self.lpm(annual_mar=0, moment=1)

        gain_loss_ratio = higher_partial_moment / lower_partial_moment

        return gain_loss_ratio

    def calmar(
        self, periods=0, inverse=True, annual_rfr=0.03, annual=True, compounding=True
    ):
        """
        Calculates Calmar ratio

        :param periods: Number of periods taken into consideration for maximum drawdown calculation, defaults to 0
        :type periods: int, optional
        :param inverse: Whether to invert (i.e. make positive) maximum drawdown, defaults to True
        :type inverse: bool, optional
        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Calmar ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        maximum_drawdown = self.maximum_drawdown(periods=periods, inverse=inverse)

        if annual and compounding:
            calmar_ratio = (self.geometric_mean - annual_rfr) / maximum_drawdown
        elif annual and not compounding:
            calmar_ratio = (self.arithmetic_mean - annual_rfr) / maximum_drawdown
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            calmar_ratio = (self.mean - rfr) / maximum_drawdown

        return calmar_ratio

    def sterling(
        self,
        annual_rfr=0.03,
        annual_excess=0.1,
        largest=0,
        inverse=True,
        annual=True,
        compounding=True,
        original=True,
    ):
        """
        Calculates Sterling ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual_excess: Annual return above average largest drawdown, defaults to 0.1
        :type annual_excess: float, optional
        :param largest: Number of largest drawdowns taken into consideration for the calculation, defaults to 0
        :type largest: int, optional
        :param inverse: Whether to invert (i.e. make positive) average drawdown, defaults to True
        :type inverse: bool, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param original: Whether to calculate the original version of Sterling ratio or Sterling-Calmar ratio, defaults to True
        :type original: bool, optional
        :return: Sterling ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        average_drawdown = self.average_drawdown(largest=largest, inverse=inverse)

        if original:
            if annual and compounding:
                sterling_ratio = self.geometric_mean / (
                    average_drawdown + annual_excess
                )
            elif annual and not compounding:
                sterling_ratio = self.arithmetic_mean / (
                    average_drawdown + annual_excess
                )
            elif not annual:
                excess = self._rate_conversion(annual_excess)
                sterling_ratio = self.mean / (average_drawdown + excess)

        else:
            if annual and compounding:
                sterling_ratio = (self.geometric_mean - annual_rfr) / average_drawdown
            elif annual and not compounding:
                sterling_ratio = (self.arithmetic_mean - annual_rfr) / average_drawdown
            elif not annual:
                rfr = self._rate_conversion(annual_rfr)
                sterling_ratio = (self.mean - rfr) / average_drawdown

        return sterling_ratio

    def ulcer(self):
        """
        Calculates Ulcer Index

        :return: Ulcer Index
        :rtype: float
        """

        ss_drawdowns = (
            (100 * (self.state[self.name] / self.state[self.name].cummax() - 1)) ** 2
        ).sum()
        ulcer_index = np.sqrt(ss_drawdowns / self.state.shape[0])

        return ulcer_index

    def martin(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
    ):
        """
        Calculates Martin ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Martin ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        ulcer_index = self.ulcer()

        if annual and compounding:
            martin_ratio = (self.geometric_mean - annual_rfr) / ulcer_index
        elif annual and not compounding:
            martin_ratio = (self.arithmetic_mean - annual_rfr) / ulcer_index
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            martin_ratio = (self.mean - rfr) / ulcer_index

        return martin_ratio

    def parametric_var(self, ci=0.95, frequency=1):
        """
        Calculates parametric Value-at-Risk (VaR)

        :param ci: Confidence interval for VaR, defaults to 0.95
        :type ci: float, optional
        :param frequency: frequency for changing periods of VaR. For example, if
                          `self.prices` contains daily data and there are 252
                          trading days in a year, setting`frequency=252` will
                          yield annual VaR, defaults to 1 (same as data)
        :type frequency: int, optional
        :return: Parametric VaR
        :rtype: float
        """

        return stats.norm.ppf(1 - ci, self.mean, self.volatility) * np.sqrt(frequency)

    def historical_var(self, ci=0.95, frequency=1):
        """
        Calculates historical Value-at-Risk (VaR)

        :param ci: Confidence interval for VaR, defaults to 0.95
        :type ci: float, optional
        :param frequency: frequency for changing periods of VaR. For example, if
                          `self.prices` contains daily data and there are 252
                          trading days in a year, setting`frequency=252` will
                          yield annual VaR, defaults to 1 (same as data)
        :type frequency: int, optional
        :return: Historical VaR
        :rtype: float
        """

        return np.percentile(self.returns, 100 * (1 - ci)) * np.sqrt(frequency)

    def plot_parametric_var(
        self,
        ci=0.95,
        frequency=1,
        plot_z=3,
        style=STYLE,
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw,
    ):
        """
        Plot parametric Value-at-Risk (VaR) model elements

        :param ci: Confidence interval for VaR, defaults to 0.95
        :type ci: float, optional
        :param frequency: frequency for changing periods of VaR. For example, if
                          `self.prices` contains daily data and there are 252
                          trading days in a year, setting`frequency=252` will
                          yield annual VaR, defaults to 1 (same as data)
        :type frequency: int, optional
        :param plot_z: Normal distribution z-value that specifies how much of the
                       distribution will be plotted, defaults to 3
        :type plot_z: int, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        var = self.parametric_var(ci, frequency)
        x = np.linspace(
            self.mean - plot_z * self.volatility,
            self.mean + plot_z * self.volatility,
            100,
        )
        pdf = stats.norm(self.mean, self.volatility).pdf(x)

        cutoff = (np.abs(x - var)).argmin()

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ax.plot(
            x,
            pdf,
            label="Analytical Distribution",
        )
        ax.fill_between(
            x[0:cutoff],
            pdf[0:cutoff],
            facecolor="r",
            label=f"Value-At-Risk (ci={ci})",
        )
        ax.legend(
            loc="upper right",
        )
        ax.set_xlabel("Return")
        ax.set_ylabel("Density of Return")
        ax.set_title("Parametric VaR Plot")
        if save:
            plt.savefig(f"{self.name}_parametric_var.png", dpi=300)
        if show:
            plt.show()

    def plot_historical_var(
        self,
        ci=0.95,
        frequency=1,
        number_of_bins=100,
        style=STYLE,
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw,
    ):
        """
        Plots historical Value-at-Risk (VaR) model elements

        :param ci: Confidence interval for VaR, defaults to 0.95
        :type ci: float, optional
        :param frequency: frequency for changing periods of VaR. For example, if
                          `self.prices` contains daily data and there are 252
                          trading days in a year, setting`frequency=252` will
                          yield annual VaR, defaults to 1 (same as data)
        :type frequency: int, optional
        :param number_of_bins: Number of histogram bins, defaults to 100
        :type number_of_bins: int, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        var = self.historical_var(ci, frequency)
        sorted_returns = np.sort(self.returns, axis=0)
        bins = np.linspace(
            sorted_returns[0],
            sorted_returns[-1],
            number_of_bins,
        )[:, 0]

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ax.hist(
            sorted_returns,
            bins,
            label="Historical Distribution",
        )
        ax.axvline(x=var, ymin=0, color="r", label=f"Value-At-Risk Cutoff (ci={ci})")
        ax.legend()
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency of Return")
        ax.set_title("Historical VaR Plot")
        if save:
            plt.savefig(f"{self.name}_historical_var.png", dpi=300)
        if show:
            plt.show()

    def correlation(self):
        """
        Calculates portfolio assets returns correlation matrix

        :return: Portfolio assets returns correlation matrix
        :rtype: np.ndarray
        """

        matrix = self.assets_returns.corr().round(5)

        return matrix

    def plot_correlation(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio assets returns correlation matrix

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.correlation()

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        sns.heatmap(matrix, vmin=-1, vmax=1, center=0, annot=True, ax=ax)
        ax.set_title("Correlation Matrix")
        if save:
            plt.savefig(f"{self.name}_correlation.png", dpi=300)
        if show:
            plt.show()

    def covariance(self, method="regular", annual=False, cov_kwargs={}):
        """
        Calculates portfolio assets returns covariance matrix

        :param method: Covariance matrix method calculation. Available are:
                       `"regular"`, `"empirical"`, `"graphical_lasso"`,
                       `"elliptic_envelope"`, `"ledoit_wolf"`, `"mcd"`,
                       `"oas"`, `"shrunk_covariance"`, defaults to "regular"
        :type method: str, optional
        :param annual: Whether to calculate the covariance on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param cov_kwargs: Keyword arguments for matrix calculation
                           methods, defaults to {}
        :type cov_kwargs: dict, optional
        :raises ValueError: If matrix calculation method is unavailable
        :return: Portfolio assets returns covariance matrix
        :rtype: np.ndarray
        """

        if method == "regular":
            matrix = self.assets_returns.cov().round(5)
        elif method == "empirical":
            matrix = (
                covariance.EmpiricalCovariance(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "graphical_lasso":
            matrix = (
                covariance.GraphicalLasso(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "elliptic_envelope":
            matrix = (
                covariance.EllipticEnvelope(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "ledoit_wolf":
            matrix = (
                covariance.LedoitWolf(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "mcd":
            matrix = (
                covariance.MinCovDet(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "oas":
            matrix = (
                covariance.OAS(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "shrunk_covariance":
            matrix = (
                covariance.ShrunkCovariance(**cov_kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        else:
            raise ValueError("Covariance matrix calculation method is unavailable")

        if annual:
            matrix = matrix * self.frequency

        return matrix

    def plot_covariance(
        self,
        method="regular",
        annual=False,
        style=STYLE,
        rcParams_update={},
        show=True,
        save=False,
        cov_kwargs={},
        **fig_kw,
    ):
        """
        Plots portfolio assets returns covariance matrix

        :param method: Covariance matrix method calculation. Available are:
                       `"regular"`, `"empirical"`, `"graphical_lasso"`,
                       `"elliptic_envelope"`, `"ledoit_wolf"`, `"mcd"`,
                       `"oas"`, `"shrunk_covariance"`, defaults to "regular"
        :type method: str, optional
        :param annual: Whether to calculate the covariance on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        :param cov_kwargs: Keyword arguments for matrix calculation
                           methods, defaults to {}
        :type cov_kwargs: dict, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.covariance(method, annual, **cov_kwargs)

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        sns.heatmap(matrix, vmin=0, center=0, annot=True, ax=ax)
        ax.set_title("Covariance Matrix")
        if save:
            plt.savefig(f"{self.name}_covariance.png", dpi=300)
        if show:
            plt.show()

    def omega_ratio(self, returns=None, annual_mar=0.03):
        """
        Calculates the Omega ratio of the portfolio.

        :param returns: Array with portfolio returns for which the omega ratio is to be calculated (if different from the object portfolio), defaults to None
        :type returns: np.ndarray, optional
        :param annual_mar: Annual Minimum Acceptable Return (MAR)., defaults to 0.03
        :type annual_mar: float, optional
        :return: Omega ratio of the portfolio
        :rtype: float
        """

        if returns is None:
            returns = self.returns

        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        excess_returns = returns - mar
        winning = excess_returns[excess_returns > 0].sum()
        losing = -(excess_returns[excess_returns <= 0].sum())

        omega = winning / losing

        if not isinstance(omega, (int, float)):
            omega = omega[0]

        return omega

    def omega_sharpe_ratio(self, annual_mar=0.03):
        """
        Calculates Omega-Sharpe ratio

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Omega-Sharpe ratio
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        upside_potential = self.upside_potential(annual_mar)
        downside_potential = self.downside_potential(annual_mar)

        omega_sharpe_ratio = (
            upside_potential - downside_potential
        ) / downside_potential

        return omega_sharpe_ratio

    def plot_omega_curve(
        self,
        returns=None,
        annual_mar_lower_bound=0,
        annual_mar_upper_bound=0.1,
        style=STYLE,
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw,
    ):
        """
        Plots Omega values across different Minimum Assets Returns (MAR)
        for single or multiple portfolios

        :param returns: Array with portfolio returns for which the omega ratio is to be calculated (if different from the object portfolio), defaults to None
        :type returns: np.ndarray, optional
        :param annual_mar_lower_bound: Lower bound for MAR that will be taken to calculate the values for curves, defaults to 0
        :type annual_mar_lower_bound: float, optional
        :param annual_mar_upper_bound: Upper bound for MAR that will be taken to calculate the values for curves, defaults to 0.1
        :type annual_mar_upper_bound: float, optional
        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        if returns is None:
            returns = self.returns

        _checks._check_omega_multiple_returns(returns=returns)
        _checks._check_plot_arguments(show=show, save=save)
        _checks._check_mar_bounds(
            annual_mar_lower_bound=annual_mar_lower_bound,
            annual_mar_upper_bound=annual_mar_upper_bound,
        )

        mar_array = np.linspace(
            annual_mar_lower_bound,
            annual_mar_upper_bound,
            round(100 * (annual_mar_upper_bound - annual_mar_lower_bound)),
        )
        all_values = pd.DataFrame(index=mar_array, columns=returns.columns)

        for portfolio in returns.columns:
            omega_values = list()
            for mar in mar_array:
                value = np.round(self.omega_ratio(returns[portfolio], mar), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        all_values.plot(ax=ax)
        ax.set_xlabel("Minimum Acceptable Return (MAR)")
        ax.set_ylabel("Omega Ratio")
        ax.set_title("Omega Curves")
        if save:
            plt.savefig("omega_curves.png", dpi=300)
        if show:
            plt.show()

    def herfindahl_index(self):
        """
        Calculates Herfindahl Index

        :return: Herfindahl Index
        :rtype: float
        """

        acf = stattools.acf(self.returns)
        positive_acf = acf[acf >= 0][1:]
        positive_acf = positive_acf[~np.isnan(positive_acf)]
        scaled_acf = positive_acf / np.sum(positive_acf)
        herfindahl_index = np.sum(scaled_acf**2)

        return herfindahl_index

    def appraisal(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Appraisal ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Appraisal ratio
        :rtype: float
        """

        capm = self.capm(annual_rfr, benchmark)

        if annual and compounding:
            specific_risk = np.sqrt(
                np.sum((capm[2] - capm[2].mean()) ** 2) / capm[2].shape[0]
            ) * np.sqrt(self.frequency)

            appraisal_ratio = (
                self.jensen_alpha(annual_rfr, annual, compounding) / specific_risk
            )
        elif annual and not compounding:
            specific_risk = np.sqrt(
                np.sum((capm[2] - capm[2].mean()) ** 2) / capm[2].shape[0]
            ) * np.sqrt(self.frequency)

            appraisal_ratio = (
                self.jensen_alpha(annual_rfr, annual, compounding) / specific_risk
            )
        elif not annual:
            specific_risk = np.sqrt(
                np.sum((capm[2] - capm[2].mean()) ** 2) / capm[2].shape[0]
            )

            appraisal_ratio = (
                self.jensen_alpha(annual_rfr, annual, compounding) / specific_risk
            )

        return appraisal_ratio[0]

    def burke(
        self,
        annual_rfr=0.03,
        largest=0,
        annual=True,
        compounding=True,
        modified=False,
        **sorted_drawdowns_kwargs,
    ):
        """
        Calculates Burke ratio

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param largest: Number of largest drawdowns taken into consideration for the calculation, defaults to 0
        :type largest: int, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param modified: Whether to calculate modified Burke ratio, defaults to False
        :type modified: bool, optional
        :return: Burke ratio
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_booleans(modified=modified)

        drawdowns = self.sorted_drawdowns(largest=largest, **sorted_drawdowns_kwargs)
        if annual and compounding:
            burke_ratio = (self.geometric_mean - annual_rfr) / np.sqrt(
                np.sum(drawdowns**2)
            )
        elif annual and not compounding:
            burke_ratio = (self.arithmetic_mean - annual_rfr) / np.sqrt(
                np.sum(drawdowns**2)
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            burke_ratio = (self.mean - rfr) / np.sqrt(np.sum(drawdowns**2))

        if modified:
            burke_ratio = burke_ratio * np.sqrt(self.returns.shape[0])

        return burke_ratio[0]

    def hurst_index(self):
        """
        Calculates Hurst Index

        :return: Hurst Index
        :rtype: float
        """

        m = (self.returns.max() - self.returns.min()) / np.std(self.returns)
        n = self.returns.shape[0]
        hurst_index = np.log(m) / np.log(n)

        return hurst_index[0]

    def bernardo_ledoit(self):
        """
        Calculates Bernardo and Ledoit ratio

        :return: Bernardo and Ledoit ratio
        :rtype: float
        """

        positive_returns = self.returns[self.returns > 0].dropna()
        negative_returns = self.returns[self.returns < 0].dropna()

        bernardo_ledoit_ratio = np.sum(positive_returns) / -np.sum(negative_returns)

        return bernardo_ledoit_ratio[0]

    def skewness_kurtosis_ratio(self):
        """
        Calculates skewness-kurtosis ratio

        :return: skewness-kurtosis ratio
        :rtype: float
        """

        skewness_kurtosis_ratio = self.skewness / self.kurtosis

        return skewness_kurtosis_ratio

    def d(self):
        """
        Calculates D ratio

        :return: D ratio
        :rtype: float
        """

        positive_returns = self.returns[self.returns > 0].dropna()
        negative_returns = self.returns[self.returns < 0].dropna()

        d_ratio = (negative_returns.shape[0] * np.sum(negative_returns)) / (
            positive_returns.shape[0] * np.sum(positive_returns)
        )

        return -d_ratio[0]

    def kelly_criterion(self, annual_rfr=0.03):
        """
        Calculates Kelly criterion

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :return: Kelly criterion
        :rtype: float
        """

        _checks._check_rate_arguments(annual_rfr=annual_rfr)

        rfr = self._rate_conversion(annual_rfr)
        excess_returns = self.returns - rfr
        kelly_criterion = excess_returns.mean() / np.var(self.returns)

        return kelly_criterion[0]

    def modigliani(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        adjusted=False,
        probabilistic=False,
        sharpe_benchmark=0.0,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Modigliani-Modigliani measure

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param adjusted: Whether to use adjusted Sharpe ratio for calculation, defaults to False
        :type adjusted: bool, optional
        :param probabilistic: Whether to use probabilistic Sharpe ratio for calculation, defaults to False
        :type probabilistic: bool, optional
        :param sharpe_benchmark: Benchmark Sharpe ratio for probabilistic Sharpe ratio (if used), defaults to 0.0
        :type sharpe_benchmark: float, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Modigliani-Modigliani measure
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        sharpe_ratio = self.sharpe(
            annual_rfr=annual_rfr,
            annual=annual,
            compounding=compounding,
            adjusted=adjusted,
            probabilistic=probabilistic,
            sharpe_benchmark=sharpe_benchmark,
        )

        if annual and compounding:
            modigliani_measure = (
                sharpe_ratio * np.std(self.benchmark_returns) * np.sqrt(self.frequency)
                + annual_rfr
            )
        elif annual and not compounding:
            modigliani_measure = (
                sharpe_ratio * np.std(self.benchmark_returns) * np.sqrt(self.frequency)
                + annual_rfr
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            modigliani_measure = sharpe_ratio * np.std(self.benchmark_returns) + rfr

        return modigliani_measure[0]

    def fama_beta(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Fama beta

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Fama beta
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        fama_beta = np.std(self.returns) / np.std(self.benchmark_returns.to_numpy())

        return fama_beta[0]

    def diversification(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Diversification measure

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Diversification
        :rtype: float
        """

        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        fama_beta = self.fama_beta(benchmark)
        capm = self.capm(annual_rfr, benchmark)

        if annual and compounding:
            diversification = (fama_beta - capm[1]) * (
                self.benchmark_geometric_mean - annual_rfr
            )
        elif annual and not compounding:
            diversification = (fama_beta - capm[1]) * (
                self.benchmark_arithmetic_mean - annual_rfr
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            diversification = (fama_beta - capm[1]) * (self.benchmark_mean - rfr)

        return diversification

    def net_selectivity(self, annual_rfr=0.03, annual=True, compounding=True):
        """
        Calculates Net Selectivity

        :param annual_rfr: Annual Risk-free Rate (RFR), defaults to 0.03
        :type annual_rfr: float, optional
        :param annual: Whether to calculate the statistic on annual basis or data frequency basis, defaults to True
        :type annual: bool, optional
        :param compounding: If `annual=True`, specifies if returns should be compounded, defaults to True
        :type compounding: bool, optional
        :return: Net Selectivity
        :rtype: float
        """

        jensen_alpha = self.jensen_alpha(annual_rfr, annual, compounding)
        diversification = self.diversification(annual_rfr, annual, compounding)

        net_selectivity = jensen_alpha - diversification

        return net_selectivity

    def downside_frequency(self, annual_mar=0.03):
        """
        Calculates Downside frequency

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Downside frequency
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        losing = self.returns[self.returns <= mar].dropna()

        downside_frequency = losing.shape[0] / self.returns.shape[0]

        return downside_frequency

    def upside_frequency(self, annual_mar=0.03):
        """
        Calculates Upside frequency

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Upside frequency
        :rtype: float
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        winning = self.returns[self.returns > mar].dropna()

        upside_frequency = winning.shape[0] / self.returns.shape[0]

        return upside_frequency

    def up_capture(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Up-market capture

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Up-market capture
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        positive_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns > 0
        ].dropna()
        corresponding_returns = self.returns.loc[positive_benchmark_returns.index]

        up_capture_indicator = (
            corresponding_returns.mean()[0] / positive_benchmark_returns.mean()[0]
        )

        return up_capture_indicator

    def down_capture(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Down-market capture

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Down-market capture
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        negative_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns <= 0
        ].dropna()
        corresponding_returns = self.returns.loc[negative_benchmark_returns.index]

        down_capture_indicator = (
            corresponding_returns.mean()[0] / negative_benchmark_returns.mean()[0]
        )

        return down_capture_indicator

    def up_number(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Up-market number

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Up-market number
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        positive_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns > 0
        ].dropna()
        corresponding_returns = self.returns.loc[positive_benchmark_returns.index][
            self.returns > 0
        ].dropna()

        up_number_ratio = (
            corresponding_returns.shape[0] / positive_benchmark_returns.shape[0]
        )

        return up_number_ratio

    def down_number(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Down-market number

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Down-market number
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        negative_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns <= 0
        ].dropna()
        corresponding_returns = self.returns.loc[negative_benchmark_returns.index][
            self.returns < 0
        ].dropna()

        down_number_ratio = (
            corresponding_returns.shape[0] / negative_benchmark_returns.shape[0]
        )

        return down_number_ratio

    def up_percentage(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Up-market percentage

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Up-market percentage
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        positive_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns > 0
        ].dropna()
        corresponding_returns = self.returns[
            (self.returns[self.name] > self.benchmark_returns[self.benchmark_name])
            & (self.benchmark_returns[self.benchmark_name] > 0)
        ].dropna()

        up_percentage = (
            corresponding_returns.shape[0] / positive_benchmark_returns.shape[0]
        )

        return up_percentage

    def down_percentage(
        self,
        benchmark={
            "benchmark_tickers": None,
            "benchmark_prices": None,
            "benchmark_weights": None,
            "benchmark_name": "Benchmark Portfolio",
            "start": "1970-01-02",
            "end": CURRENT_DATE,
            "interval": "1d",
        },
    ):
        """
        Calculates Down-market percentage

        :param benchmark: Benchmark details that can be provided to set or reset (i.e. change) benchmark portfolio, defaults to { "benchmark_tickers": None, "benchmark_prices": None, "benchmark_weights": None, "benchmark_name": "Benchmark Portfolio", "start": "1970-01-02", "end": CURRENT_DATE, "interval": "1d", }
        :type benchmark: dict, optional
        :return: Down-market percentage
        :rtype: float
        """

        set_benchmark = _checks._whether_to_set(
            slf_benchmark_prices=self.benchmark_prices, **benchmark
        )
        if set_benchmark:
            self._set_benchmark(**benchmark)

        negative_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns <= 0
        ].dropna()
        corresponding_returns = self.returns[
            (self.returns[self.name] > self.benchmark_returns[self.benchmark_name])
            & (self.benchmark_returns[self.benchmark_name] <= 0)
        ].dropna()

        down_percentage = (
            corresponding_returns.shape[0] / negative_benchmark_returns.shape[0]
        )

        return down_percentage

    def drawdowns(self):
        """
        Calculates portfolio drawdown level at each time-point

        :return: Drawdown levels
        :rtype: np.ndarray
        """

        cumulative_concatenated = pd.concat(
            [
                pd.DataFrame([1], columns=[self.name]),
                self.cumulative_returns,
            ]
        ).rename(index={0: self.prices.index[0]})
        cummax = cumulative_concatenated.cummax()

        drawdowns = cumulative_concatenated / cummax - 1

        return drawdowns

    def maximum_drawdown(self, periods=0, inverse=True):
        """
        Calculates Maximum drawdown

        :param periods: Number of periods taken into consideration for maximum drawdown calculation, defaults to 0
        :type periods: int, optional
        :param inverse: Whether to invert (i.e. make positive) maximum drawdown, defaults to True
        :type inverse: bool, optional
        :return: Maximum drawdown
        :rtype: float
        """

        _checks._check_periods(periods=periods, state=self.state)
        _checks._check_booleans(inverse=inverse)

        drawdowns = self.drawdowns()

        if inverse:
            mdd = -drawdowns[-periods:].min()[0]
        else:
            mdd = drawdowns[-periods:].min()[0]

        return mdd

    def average_drawdown(self, largest=0, inverse=True):
        """
        Calculates Average drawdown

        :param largest: Number of largest drawdowns taken into consideration for the calculation, defaults to 0
        :type largest: int, optional
        :param inverse: Whether to invert (i.e. make positive) average drawdown, defaults to True
        :type inverse: bool, optional
        :return: Average drawdown
        :rtype: float
        """

        _checks._check_nonnegints(largest=largest)
        _checks._check_booleans(inverse=inverse)

        drawdowns = self.drawdowns()
        drawdowns = drawdowns.sort_values(by=self.name, ascending=False)[-largest:]

        if inverse:
            add = -drawdowns.mean()[0]
        else:
            add = drawdowns.mean()[0]

        return add

    def sorted_drawdowns(self, largest=0, **sorted_drawdowns_kwargs):
        """
        Sorts the portfolio drawdowns at each time-point

        :param largest: Number of largest drawdowns returned, defaults to 0
        :type largest: int, optional
        :return: Sorted drawdowns
        :rtype: float
        """

        _checks._check_nonnegints(largest=largest)

        drawdowns = self.drawdowns()
        sorted_drawdowns = drawdowns.sort_values(
            by=self.name, ascending=False, **sorted_drawdowns_kwargs
        )[-largest:]

        return sorted_drawdowns

    def plot_drawdowns(
        self, style=STYLE, rcParams_update={}, show=True, save=False, **fig_kw
    ):
        """
        Plots portfolio drawdowns

        :param style: `matplotlib` style to be used for plots. User can pass
                      built-in `matplotlib` style (e.g. `classic`, `fivethirtyeight`),
                      or a path to a custom style defined in a `.mplstyle` document,
                      defaults to STYLE (propriatery PortAn style)
        :type style: str, optional
        :param rcParams_update: `matplotlib.rcParams` to modify the style defined by
                                `style` argument, defaults to {} (no modification)
        :type rcParams_update: dict, optional
        :param show: Whether to show the plot, defaults to True
        :type show: bool, optional
        :param save: Whether to save the plot as `.png` file, defaults to False
        :type save: bool, optional
        """

        _checks._check_plot_arguments(show=show, save=save)

        drawdowns = self.drawdowns()

        plt.style.use(style)
        plt.rcParams.update(**rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        drawdowns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title("Portfolio Drawdowns")
        if save:
            plt.savefig(f"{self.name}_drawdowns.png", dpi=300)
        if show:
            plt.show()

    def upside_risk(self, annual_mar=0.03):
        """
        Calculates Upside risk (also referred to as Upside semideviation)

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Upside risk
        :rtype: float
        """

        upside_risk = np.sqrt(self.hpm(annual_mar=annual_mar, moment=2))

        return upside_risk

    def upside_potential(self, annual_mar=0.03):
        """
        Calculates Upside potential

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Upside potential
        :rtype: float
        """

        upside_potential = self.hpm(annual_mar=annual_mar, moment=1)

        return upside_potential

    def upside_variance(self, annual_mar=0.03):
        """
        Calculates Upside variance

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Upside variance
        :rtype: float
        """

        upside_variance = self.hpm(annual_mar=annual_mar, moment=2)

        return upside_variance

    def downside_risk(self, annual_mar=0.03):
        """
        Calculates Downside risk (also referred to as Downside semideviation)

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Downside risk
        :rtype: float
        """

        downside_risk = np.sqrt(self.lpm(annual_mar=annual_mar, moment=2))

        return downside_risk

    def downside_potential(self, annual_mar=0.03):
        """
        Calculates Downside potential

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Downside potential
        :rtype: float
        """

        downside_potential = self.lpm(annual_mar=annual_mar, moment=1)

        return downside_potential

    def downside_variance(self, annual_mar=0.03):
        """
        Calculates Downside variance

        :param annual_mar: Annual Minimum Accepted Return (MAR), defaults to 0.03
        :type annual_mar: float, optional
        :return: Downside variance
        :rtype: float
        """

        downside_variance = self.lpm(annual_mar=annual_mar, moment=2)

        return downside_variance
