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
import warnings
from portfolio_analytics import _checks


class PortfolioAnalytics:
    def __init__(
        self,
        prices,
        weights,
        benchmark_prices=None,
        benchmark_weights=None,
        name="Investment Portfolio",
        benchmark_name="Benchmark Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        prices, weights, benchmark_prices, benchmark_weights = _checks._check_init(
            prices=prices,
            weights=weights,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            name=name,
            benchmark_name=benchmark_name,
            initial_aum=initial_aum,
            frequency=frequency,
        )

        self.prices = prices
        self.assets_returns = self.prices.pct_change().drop(self.prices.index[0])
        self.tickers = self.prices.columns
        self.weights = weights
        self.assets_info = np.empty(len(self.tickers), dtype=object)
        self.assets_names = np.empty(len(self.tickers), dtype="<U64")
        for i, ticker in enumerate(self.tickers):
            self.assets_info[i] = yf.Ticker(ticker).info
            self.assets_names[i] = self.assets_info[i]["longName"]
        self.name = name
        self.initial_aum = initial_aum
        self.frequency = frequency

        # funds allocated to each asset
        self.allocation_funds = pd.Series(
            np.multiply(self.initial_aum, self.weights), index=self.tickers
        )

        # number of assets bought at t0
        self.allocation_assets = pd.Series(
            np.divide(self.allocation_funds, self.prices.iloc[0].T), index=self.tickers
        )

        # absolute (dollar) value of each asset in portfolio (i.e. state of the portfolio, not rebalanced)
        self.state = pd.DataFrame(
            np.multiply(self.prices, self.allocation_assets),
            index=self.prices.index,
            columns=self.tickers,
        )
        self.state["Portfolio"] = self.state.sum(axis=1)

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

        self.mean = self.returns.mean(axis=0)[0]
        self.arithmetic_mean = self.mean * self.frequency
        self.geometric_mean = (
            (1 + self.returns).prod() ** (self.frequency / self.returns.shape[0]) - 1
        )[0]

        self.volatility = self.returns.std()[0]
        self.annual_volatility = self.volatility * np.sqrt(self.frequency)

        self.skewness = stats.skew(self.returns)[0]
        self.kurtosis = stats.kurtosis(self.returns)[0]

        self.min_aum = self.state["Portfolio"].min()
        self.max_aum = self.state["Portfolio"].max()
        self.mean_aum = self.state["Portfolio"].mean(axis=0)
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
            self.benchmark_weights = benchmark_weights
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

            self.benchmark_mean = self.benchmark_returns.mean(axis=0)[0]
            self.benchmark_arithmetic_mean = self.benchmark_mean * self.frequency
            self.benchmark_geometric_mean = (
                (1 + self.benchmark_returns).prod()
                ** (self.frequency / self.benchmark_returns.shape[0])
                - 1
            )[0]

    def set_benchmark(self, benchmark_prices, benchmark_weights, benchmark_name):
        self.benchmark_prices = benchmark_prices
        self.benchmark_weights = benchmark_weights
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

        self.benchmark_mean = self.benchmark_returns.mean(axis=0)[0]
        self.benchmark_arithmetic_mean = self.benchmark_mean * self.frequency
        self.benchmark_geometric_mean = (
            (1 + self.benchmark_returns).prod()
            ** (self.frequency / self.benchmark_returns.shape[0])
            - 1
        )[0]

        warnings.warn()

    def _rate_conversion(self, annual_rate):
        return (annual_rate + 1) ** (1 / self.frequency) - 1

    def excess_return_above_mar(self, annual_mar=0.03, annual=True, compounding=True):
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
        _checks._check_percentage(percentage=percentage)

        final_aum = self.final_aum

        if not percentage:
            net_return = final_aum - self.initial_aum
        else:
            net_return = (final_aum - self.initial_aum) / self.initial_aum

        return net_return

    def distribution_test(self, test="dagostino-pearson", distribution="norm"):
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

    def ewm_return(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        annual=True,
        compounding=True,
        **kwargs
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)

        if annual and compounding:
            mean = (
                1
                + self.returns.ewm(com, span, halflife, alpha, **kwargs).mean(axis=0).iloc[-1]
            ) ** self.frequency - 1
        elif annual and not compounding:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **kwargs).mean(axis=0).iloc[-1]
                * self.frequency
            )
        elif not annual:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **kwargs).mean(axis=0).iloc[-1]
            )

        return mean[0]

    def plot_aum(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.state["Portfolio"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("AUM ($)")
        ax.set_title("Assets Under Management")
        if save:
            plt.savefig("aum.png", dpi=300)
        if show:
            plt.show()

    def plot_returns(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.set_title("Portfolio Returns")
        if save:
            plt.savefig("returns.png", dpi=300)
        if show:
            plt.show()

    def plot_returns_distribution(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.returns.plot.hist(ax=ax, bins=90)
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency")
        ax.set_title("Portfolio Returns Distribution")
        if save:
            plt.savefig("returns_distribution.png", dpi=300)
        if show:
            plt.show()

    def plot_cumulative_returns(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Portfolio Cumulative Returns")
        if save:
            plt.savefig("cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_piechart(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        wp = {"linewidth": 1, "edgecolor": "black"}
        explode = tuple(repeat(0.05, len(self.tickers)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
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
        ax.set_title(str(self.name + " Asset Distribution"))
        if save:
            plt.savefig(str(self.name + "_pie_chart.png"), dpi=300)
        if show:
            plt.show()

    def plot_assets_cumulative_returns(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        assets_cumulative_returns = (self.assets_returns + 1).cumprod()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        assets_cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Assets Cumulative Returns")
        ax.legend(labels=self.assets_names)
        if save:
            plt.savefig("assets_cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def _ap(self, pct, all_values):
        absolute = int(pct / 100.0 * np.sum(all_values))

        return "{:.1f}%\n(${:d})".format(pct, absolute)

    def capm_return(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        capm = self.capm(
            annual_rfr=annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        if annual and compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_geometric_mean - annual_rfr)
        elif annual and not compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_arithmetic_mean - annual_rfr)
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            mean = rfr + capm[1] * (self.benchmark_mean - rfr)

        return mean

    def capm(
        self,
        annual_rfr=0.02,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        rfr = self._rate_conversion(annual_rfr)

        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(excess_benchmark_returns, excess_returns)
        alpha = model.intercept_[0]
        beta = model.coef_[0][0]
        r_squared = model.score(excess_benchmark_returns, excess_returns)
        epsilon = np.subtract(
            excess_returns.to_numpy(), alpha - beta * excess_benchmark_returns
        )

        return (
            alpha,
            beta,
            r_squared,
            epsilon,
            excess_returns,
            excess_benchmark_returns,
        )

    def plot_capm(
        self,
        annual_rfr=0.02,
        show=True,
        save=False,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_plot_arguments(show=show, save=save)

        capm = self.capm(
            annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(capm[5], capm[4], color="b")
        ax.plot(capm[5], capm[0] + capm[1] * capm[5], color="r")
        empty_patch = mpatches.Patch(color="none", visible=False)
        ax.legend(
            handles=[empty_patch, empty_patch],
            labels=[
                r"$\alpha$" + " = " + str(np.round(capm[0], 5)),
                r"$\beta$" + " = " + str(np.round(capm[1], 5)),
            ],
        )
        ax.set_xlabel("Benchmark Excess Returns")
        ax.set_ylabel("Portfolio Excess Returns")
        ax.set_title("Portfolio Excess Returns Against Benchmark (CAPM)")
        if save:
            plt.savefig("capm.png", dpi=300)
        if show:
            plt.show()

    def sharpe(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        adjusted=False,
        probabilistic=False,
        sharpe_benchmark=0.0,
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_sharpe(adjusted=adjusted, probabilistic=probabilistic)

        if annual and compounding:
            sharpe_ratio = (
                100 * (self.geometric_mean - annual_rfr) / self.annual_volatility
            )
        elif annual and not compounding:
            sharpe_ratio = (
                100 * (self.arithmetic_mean - annual_rfr) / self.annual_volatility
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            sharpe_ratio = 100 * (self.mean - rfr) / self.volatility

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

    def excess_return(
        self,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        if annual and compounding:
            excess_return = self.geometric_mean - self.benchmark_geometric_mean
        elif annual and not compounding:
            excess_return = self.arithmetic_mean - self.benchmark_arithmetic_mean
        elif not annual:
            excess_return = self.mean - self.benchmark_mean

        return excess_return

    def tracking_error(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        tracking_error = np.std(self.returns - self.benchmark_returns.to_numpy())

        return tracking_error[0]

    def information_ratio(
        self,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        excess_return = self.excess_return(
            annual=annual,
            compounding=compounding,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        tracking_error = self.tracking_error(
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        information_ratio = excess_return / tracking_error

        return information_ratio

    def upside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        positive_returns = self.returns - mar
        positive_returns = positive_returns[positive_returns > 0]
        if annual:
            upside_volatility = np.std(positive_returns, ddof=1) * self.frequency
        else:
            upside_volatility = np.std(positive_returns, ddof=1)

        return upside_volatility[0]

    def downside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        negative_returns = self.returns - mar
        negative_returns = negative_returns[negative_returns < 0]
        if annual:
            downside_volatility = np.std(negative_returns, ddof=1) * self.frequency
        else:
            downside_volatility = np.std(negative_returns, ddof=1)

        return downside_volatility[0]

    def volatility_skewness(self, annual_mar=0.03, annual=True):
        upside = self.upside_volatility(annual_mar, annual)
        downside = self.downside_volatility(annual_mar, annual)
        skew = upside / downside

        return skew

    def omega_excess_return(
        self,
        annual_mar=0.03,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        portfolio_downside_volatility = self.downside_volatility(annual_mar, annual)

        mar = self._rate_conversion(annual_mar)

        negative_benchmark_returns = self.benchmark_returns - mar
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]
        if annual:
            benchmark_downside_volatility = (
                np.std(negative_benchmark_returns, ddof=1) * self.frequency
            )
        else:
            benchmark_downside_volatility = np.std(negative_benchmark_returns, ddof=1)

        if annual and compounding:
            omega_excess_return = (
                self.geometric_mean
                - 3 * portfolio_downside_volatility * benchmark_downside_volatility
            )
        elif annual and not compounding:
            omega_excess_return = (
                self.arithmetic_mean
                - 3 * portfolio_downside_volatility * benchmark_downside_volatility
            )
        elif not annual:
            omega_excess_return = (
                self.mean
                - 3 * portfolio_downside_volatility * benchmark_downside_volatility
            )

        return omega_excess_return[0]

    def upside_potential(self, annual_mar=0.03, annual=True):
        mar = self._rate_conversion(annual_mar)

        downside_volatility = self.downside_volatility(annual_mar, annual)
        upside = self.returns - mar
        upside = upside[upside > 0].sum()
        upside_potential_ratio = upside / downside_volatility

        return upside_potential_ratio[0]

    def downside_volatility_ratio(
        self,
        annual_mar=0.03,
        annual=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        portfolio_downside_volatility = self.downside_volatility(annual_mar, annual)

        mar = self._rate_conversion(annual_mar)

        negative_benchmark_returns = self.benchmark_returns - mar
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]
        if annual:
            benchmark_downside_volatility = (
                np.std(negative_benchmark_returns, ddof=1) * self.frequency
            )
        else:
            benchmark_downside_volatility = np.std(negative_benchmark_returns, ddof=1)

        downside_volatility_ratio = (
            portfolio_downside_volatility / benchmark_downside_volatility
        )

        return downside_volatility_ratio[0]

    def sortino(self, annual_mar=0.03, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_mar=annual_mar,
            annual_rfr=annual_rfr,
            annual=annual,
            compounding=compounding,
        )

        downside_volatility = self.downside_volatility(annual_mar, annual)

        if annual and compounding:
            sortino_ratio = (
                100 * (self.geometric_mean - annual_rfr) / downside_volatility
            )
        elif annual and not compounding:
            sortino_ratio = (
                100 * (self.arithmetic_mean - annual_rfr) / downside_volatility
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            sortino_ratio = 100 * (self.mean - rfr) / downside_volatility

        return sortino_ratio

    def jensen_alpha(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        rfr = self._rate_conversion(annual_rfr)

        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(excess_benchmark_returns, excess_returns)
        beta = model.coef_[0]

        if annual and compounding:
            jensen_alpha = (
                self.geometric_mean
                - annual_rfr
                - beta * (self.benchmark_geometric_mean - annual_rfr)
            )
        elif annual and not compounding:
            jensen_alpha = (
                self.arithmetic_mean
                - annual_rfr
                - beta * (self.benchmark_arithmetic_mean - annual_rfr)
            )
        elif not annual:
            jensen_alpha = self.mean - rfr - beta * (self.benchmark_mean - rfr)

        return jensen_alpha[0]

    def treynor(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        rfr = self._rate_conversion(annual_rfr)

        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(excess_benchmark_returns, excess_returns)
        beta = model.coef_[0]

        if annual and compounding:
            treynor_ratio = 100 * (self.geometric_mean - annual_rfr) / beta
        elif annual and not compounding:
            treynor_ratio = 100 * (self.arithmetic_mean - annual_rfr) / beta
        elif not annual:
            treynor_ratio = 100 * (self.mean - rfr) / beta

        return treynor_ratio[0]

    def hpm(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)

        days = self.returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.max(self.returns - mar, 0), moment)
        )

        return higher_partial_moment

    def lpm(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.max(mar - self.returns, 0), moment)
        )

        return lower_partial_moment

    def kappa(self, annual_mar=0.03, moment=3, annual=True, compounding=True):
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
            kappa_ratio = (
                100 * (self.mean - mar) / np.power(lower_partial_moment, (1 / moment))
            )

        return kappa_ratio

    def gain_loss(self, annual_mar=0.03, moment=1):
        higher_partial_moment = self.hpm(annual_mar, moment)
        lower_partial_moment = self.lpm(annual_mar, moment)

        gain_loss_ratio = higher_partial_moment / lower_partial_moment

        return gain_loss_ratio

    def calmar(self, periods=1000, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        maximum_drawdown = self.maximum_drawdown()

        if annual and compounding:
            calmar_ratio = 100 * (self.geometric_mean - annual_rfr) / maximum_drawdown
        elif annual and not compounding:
            calmar_ratio = 100 * (self.arithmetic_mean - annual_rfr) / maximum_drawdown
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            calmar_ratio = 100 * (self.mean - rfr) / maximum_drawdown

        return calmar_ratio

    def sterling(self, annual_rfr=0.02, drawdowns=3, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_posints(drawdowns=drawdowns)

        portfolio_drawdowns = self.drawdowns()
        sorted_drawdowns = np.sort(portfolio_drawdowns)
        d_average_drawdown = np.mean(sorted_drawdowns[-drawdowns:])

        if annual and compounding:
            sterling_ratio = (
                100 * (self.geometric_mean - annual_rfr) / np.abs(d_average_drawdown)
            )
        elif annual and not compounding:
            sterling_ratio = (
                100 * (self.arithmetic_mean - annual_rfr) / np.abs(d_average_drawdown)
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            sterling_ratio = 100 * (self.mean - rfr) / np.abs(d_average_drawdown)

        return sterling_ratio

    def ulcer(self, periods=14, start=1):
        periods = _checks._check_periods(periods=periods, state=self.state)
        _checks._check_posints(start=start)

        close = np.empty(periods)
        percentage_drawdown = np.empty(periods)

        if start == 1:
            periods_high = np.max(self.state.iloc[-periods:]["Portfolio"])
        else:
            periods_high = np.max(
                self.state.iloc[-periods - start + 1 : -start + 1]["Portfolio"]
            )

        for i in range(periods):
            close[i] = self.state.iloc[-i - start + 1]["Portfolio"]
            percentage_drawdown[i] = 100 * ((close[i] - periods_high)) / periods_high

        ulcer_index = np.sqrt(np.mean(np.square(percentage_drawdown)))

        return ulcer_index

    def martin(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        periods=14,
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        periods = _checks._check_periods(periods=periods, state=self.state)

        ulcer_index = self.ulcer(periods)

        if annual and compounding:
            martin_ratio = 100 * (self.geometric_mean - annual_rfr) / ulcer_index
        elif annual and not compounding:
            martin_ratio = 100 * (self.arithmetic_mean - annual_rfr) / ulcer_index
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            martin_ratio = 100 * (self.mean - rfr) / ulcer_index

        return martin_ratio

    def ulcer_series(self, periods=14):
        periods = _checks._check_periods(periods=periods, state=self.state)

        ulcer_series = pd.DataFrame(columns=["Ulcer Index"], index=self.state.index)
        for i in range(self.state.shape[0] - periods):
            ulcer_series.iloc[-i]["Ulcer Index"] = self.ulcer(periods, start=i)

        return ulcer_series

    def plot_ulcer(self, periods=14, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        ulcer_series = self.ulcer_series(periods)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ulcer_series["Ulcer Index"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ulcer Index")
        ax.set_title("Portfolio Ulcer Index")
        if save:
            plt.savefig("ulcer.png", dpi=300)
        if show:
            plt.show()

    def analytical_var(
        self, value, dof, annual=True, compounding=True, distribution="normal"
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)

        if annual and compounding:
            mean_return = self.geometric_mean
        elif annual and not compounding:
            mean_return = self.arithmetic_mean
        elif not annual:
            mean_return = self.mean

        if distribution == "normal":
            var = stats.norm(mean_return, self.annual_volatility).cdf(value)
            expected_loss = (
                stats.norm(mean_return, self.annual_volatility).pdf(
                    stats.norm(mean_return, self.annual_volatility).ppf((1 - var))
                )
                * self.annual_volatility
            ) / (1 - var) - mean_return
        elif distribution == "t":
            var = stats.t(dof).cdf(value)
            percent_point_function = stats.t(dof).ppf((1 - var))
            expected_loss = (
                -1
                / (1 - var)
                * (1 - dof) ** (-1)
                * (dof - 2 + percent_point_function**2)
                * stats.t(dof).pdf(percent_point_function)
                * self.annual_volatility
                - mean_return
            )
        else:
            raise ValueError("Probability distribution unavailable.")

        return var, expected_loss

    def historical_var(self, value):
        returns_below_value = self.returns[self.returns < value]
        var = returns_below_value.shape[0] / self.returns.shape[0]

        return var

    def plot_analytical_var(
        self,
        value,
        dof,
        annual=True,
        compounding=True,
        z=3,
        distribution="normal",
        show=True,
        save=False,
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)
        _checks._check_plot_arguments(show=show, save=save)

        if annual and compounding:
            mean_return = self.geometric_mean
        elif annual and not compounding:
            mean_return = self.arithmetic_mean
        elif not annual:
            mean_return = self.mean

        x = np.linspace(
            mean_return - z * self.annual_volatility,
            mean_return + z * self.annual_volatility,
            100,
        )

        if distribution == "normal":
            pdf = stats.norm(mean_return, self.annual_volatility).pdf(x)
        elif distribution == "t":
            pdf = stats.t(dof).pdf(x)
        else:
            raise ValueError("Probability distribution unavailable.")

        cutoff = (np.abs(x - value)).argmin()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(
            x,
            pdf,
            linewidth=2,
            color="b",
            label="Analytical Distribution",
        )
        ax.fill_between(
            x[0:cutoff], pdf[0:cutoff], facecolor="r", label="Value-At-Risk"
        )
        ax.legend(
            loc="upper right",
        )
        ax.set_xlabel("Returns")
        ax.set_ylabel("Density of Returns")
        ax.set_title(
            "Analytical (Theoretical, "
            + distribution
            + ") Return Distribution and VaR Plot"
        )
        if save:
            plt.savefig("analytical_var.png", dpi=300)
        if show:
            plt.show()

    def plot_historical_var(self, value, number_of_bins=100, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        sorted_returns = np.sort(self.returns)
        bins = np.linspace(
            sorted_returns[0],
            sorted_returns[-1] + 1,
            number_of_bins,
        )[:, 0]

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.hist(
            sorted_returns,
            bins,
            label="Historical Distribution",
        )
        ax.axvline(x=value, ymin=0, color="r", label="Value-At-Risk Cutoff")
        ax.legend()
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency of Returns")
        ax.set_title("Historical Return Distribution and VaR Plot")
        if save:
            plt.savefig("historical_var.png", dpi=300)
        if show:
            plt.show()

    def correlation(self):
        matrix = self.assets_returns.corr().round(5)

        return matrix

    def plot_correlation(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.correlation()

        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
        if save:
            plt.savefig("correlation.png", dpi=300)
        if show:
            plt.show()

    def covariance(self, method="regular", annual=False, **kwargs):
        if method == "regular":
            matrix = self.assets_returns.cov().round(5)
        elif method == "empirical":
            matrix = (
                covariance.EmpiricalCovariance(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "graphical_lasso":
            matrix = (
                covariance.GraphicalLasso(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "elliptic_envelope":
            matrix = (
                covariance.EllipticEnvelope(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "ledoit_wolf":
            matrix = (
                covariance.LedoitWolf(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )

        elif method == "mcd":
            matrix = (
                covariance.MinCovDet(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        elif method == "oas":
            matrix = (
                covariance.OAS(**kwargs).fit(self.assets_returns).covariance_.round(5)
            )
        elif method == "shrunk_covariance":
            matrix = (
                covariance.ShrunkCovariance(**kwargs)
                .fit(self.assets_returns)
                .covariance_.round(5)
            )
        else:
            raise ValueError()

        if annual:
            matrix = matrix * self.frequency

        return matrix

    def plot_covariance(
        self, method="regular", annual=False, show=True, save=False, **kwargs
    ):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.covariance(method, annual, **kwargs)

        sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
        if save:
            plt.savefig("covariance.png", dpi=300)
        if show:
            plt.show()

    def omega_ratio(self, returns=None, annual_mar=0.03):
        """
        Calculates the Omega ratio of the portfolio.

        :param annual_mar: Annual Minimum Acceptable Return (MAR)., defaults to 0.03
        :type annual_mar: float, optional
        :return: Omega ratio of the portfolio
        :rtype: pd.DataFrame
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

    def omega_sharpe_ratio(self, annual_mar=0.03, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_mar=annual_mar, annual=annual, compounding=compounding
        )

        mar = self._rate_conversion(annual_mar)
        excess_returns = self.returns - mar
        losing = (1 / excess_returns.shape[0]) * (
            -(excess_returns[excess_returns <= 0].sum())
        )

        if annual and compounding:
            annual_losing = self._rate_conversion(losing)
            omega_sharpe_ratio = (self.geometric_mean - annual_mar) / annual_losing
        elif annual and not compounding:
            annual_losing = self._rate_conversion(losing)
            omega_sharpe_ratio = (self.arithmetic_mean - annual_mar) / annual_losing
        elif not annual:
            omega_sharpe_ratio = (self.mean - mar) / losing

        return omega_sharpe_ratio[0]

    def plot_omega_curve(
        self,
        returns=None,
        annual_mar_lower_bound=0,
        annual_mar_upper_bound=0.1,
        show=True,
        save=False,
    ):
        if returns is None:
            returns = self.returns

        _checks._check_multiple_returns(returns=returns)
        _checks._check_plot_arguments(show=show, save=save)
        _checks._check_mar_bounds(
            annual_mar_lower_bound=annual_mar_lower_bound,
            annual_mar_upper_bound=annual_mar_upper_bound,
        )

        all_values = pd.DataFrame(columns=returns.columns)
        mar_array = np.linspace(
            annual_mar_lower_bound,
            annual_mar_upper_bound,
            round(100 * (annual_mar_upper_bound - annual_mar_lower_bound)),
        )

        for portfolio in returns.columns:
            omega_values = list()
            for mar in mar_array:
                value = np.round(self.omega_ratio(returns[portfolio], mar), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(
            title="Omega Curve",
            xlabel="Minimum Acceptable Return (%)",
            ylabel="Omega Ratio",
        )
        if save:
            plt.savefig("omega_curves.png", dpi=300)
        if show:
            plt.show()

    def herfindahl_index(self):
        acf = stattools.acf(self.returns)
        positive_acf = acf[acf >= 0][1:]
        positive_acf = positive_acf[~np.isnan(positive_acf)]
        scaled_acf = positive_acf / np.sum(positive_acf)
        herfindahl_index = np.sum(scaled_acf**2)

        return herfindahl_index

    def appraisal(self, annual_rfr=0.02, annual=True, compounding=True):
        capm = self.capm(annual_rfr=annual_rfr)
        specific_risk = np.sqrt(
            np.sum((capm[3] - capm[3].mean(axis=0)) ** 2) / capm[3].shape[0]
        ) * np.sqrt(self.returns.shape[0] - 1)
        appraisal_ratio = (
            self.jensen_alpha(annual_rfr, annual, compounding) / specific_risk
        )

        return appraisal_ratio[0]

    def burke(self, annual_rfr=0.02, annual=True, compounding=True, modified=False):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_booleans(argument=modified)

        drawdowns = self.drawdowns()
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
        m = (self.returns.max() - self.returns.min()) / np.std(self.returns)
        n = self.returns.shape[0]
        hurst_index = np.log(m) / np.log(n)

        return hurst_index[0]

    def bernardo_ledoit(self):
        positive_returns = self.returns[self.returns > 0].dropna()
        negative_returns = self.returns[self.returns < 0].dropna()

        bernardo_ledoit_ratio = np.sum(positive_returns) / -np.sum(negative_returns)

        return bernardo_ledoit_ratio[0]

    def skewness_kurtosis_ratio(self):
        skewness_kurtosis_ratio = self.skewness / self.kurtosis

        return skewness_kurtosis_ratio

    def d(self):
        positive_returns = self.returns[self.returns > 0].dropna()
        negative_returns = self.returns[self.returns < 0].dropna()

        d_ratio = (negative_returns.shape[0] * np.sum(negative_returns)) / (
            positive_returns.shape[0] * np.sum(positive_returns)
        )

        return d_ratio[0]

    def kelly_criterion(self, annual_rfr=0.02, half=False):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)
        _checks._check_booleans(argument=half)

        excess_returns = self.returns - annual_rfr
        kelly_criterion = excess_returns.mean(axis=0) / np.std(self.returns)

        if half:
            kelly_criterion = kelly_criterion / 2

        return kelly_criterion[0]

    def modigliani(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        adjusted=False,
        probabilistic=False,
        sharpe_benchmark=0.0,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

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
                sharpe_ratio * self.benchmark_geometric_mean + annual_rfr
            )
        elif annual and not compounding:
            modigliani_measure = (
                sharpe_ratio * self.benchmark_arithmetic_mean + annual_rfr
            )
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            modigliani_measure = sharpe_ratio * self.benchmark_mean + rfr

        return modigliani_measure

    def fama_beta(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        fama_beta = np.var(self.returns) - np.var(self.benchmark_returns.to_numpy())

        return fama_beta[0]

    def diversification(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        fama_beta = self.fama_beta()
        capm = self.capm(
            annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

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

    def net_selectivity(self, annual_rfr=0.02, annual=True, compounding=True):
        jensen_alpha = self.jensen_alpha(annual_rfr, annual, compounding)
        diversification = self.diversification(annual_rfr, annual, compounding)

        net_selectivity = jensen_alpha - diversification

        return net_selectivity

    def downside_frequency(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        excess_returns = self.returns - mar
        losing = excess_returns[excess_returns <= 0].dropna()

        downside_frequency = losing.shape[0] / self.returns.shape[0]

        return downside_frequency

    def upside_frequency(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        excess_returns = self.returns - mar
        winning = excess_returns[excess_returns > 0].dropna()

        upside_frequency = winning.shape[0] / self.returns.shape[0]

        return upside_frequency

    def upside_potential_ratio(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        mar = self._rate_conversion(annual_mar)

        downside_volatility = self.downside_volatility(annual_mar, annual)
        upside = self.returns - mar
        upside = upside[upside > 0].sum()
        upside_potential_ratio = upside / downside_volatility

        return upside_potential_ratio[0]

    def up_capture(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        positive_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns > 0
        ].dropna()
        corresponding_returns = self.returns.loc[positive_benchmark_returns.index]

        up_capture_indicator = (
            corresponding_returns.mean(axis=0)[0] / positive_benchmark_returns.mean(axis=0)[0]
        )

        return up_capture_indicator

    def down_capture(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        negative_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns <= 0
        ].dropna()
        corresponding_returns = self.returns.loc[negative_benchmark_returns.index]

        down_capture_indicator = (
            corresponding_returns.mean(axis=0)[0] / negative_benchmark_returns.mean(axis=0)[0]
        )

        return down_capture_indicator

    def up_number(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        if benchmark_prices is not None and benchmark_weights is not None:
            self.set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)
        elif benchmark_prices is not None and self.benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        elif benchmark_prices is None and self.benchmark_prices is None:
            raise ValueError(
                "Benchmark is not set. Provide benchmark prices and benchmark weights"
            )

        negative_benchmark_returns = self.benchmark_returns[
            self.benchmark_returns <= 0
        ].dropna()
        corresponding_returns = self.returns[(
            self.returns[self.name] > self.benchmark_returns[self.benchmark_name])&
        (self.benchmark_returns[self.benchmark_name] <= 0)].dropna()

        down_percentage = (
            corresponding_returns.shape[0] / negative_benchmark_returns.shape[0]
        )

        return down_percentage

    def drawdowns(self):
        cumulative_concatenated = pd.concat(
            [
                pd.DataFrame([1], columns=[self.name]),
                self.cumulative_returns,
            ]
        ).rename(index={0: self.prices.index[0]})
        cummax = cumulative_concatenated.cummax()

        drawdowns = cumulative_concatenated / cummax - 1

        return drawdowns

    def largest_individual_drawdown(self):
        drawdowns = self.drawdowns()

        return -drawdowns.min()[0]

    def maximum_drawdown(self):
        latest_high_idx = self.state["Portfolio"].idxmax()
        trough_idx = self.state["Portfolio"][:latest_high_idx].idxmin()
        peak_value = max(self.state["Portfolio"][:trough_idx])
        trough_value = self.state["Portfolio"][trough_idx]

        maximum_drawdown = (trough_value - peak_value) / peak_value

        return -maximum_drawdown
