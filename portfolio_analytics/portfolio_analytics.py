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

            self.benchmark_mean = self.benchmark_returns.mean()[0]
            self.benchmark_arithmetic_mean = self.benchmark_mean * self.frequency
            self.benchmark_geometric_mean = (
                (1 + self.benchmark_returns).prod()
                ** (self.frequency / self.benchmark_returns.shape[0])
                - 1
            )[0]

    def _set_benchmark(self, benchmark_prices, benchmark_weights, benchmark_name):
        self.benchmark_prices = benchmark_prices
        self.benchmark_weights = benchmark_weights
        self.benchmark_name = benchmark_name

        if self.prices.shape[0] != benchmark_prices.shape[0]:
            raise ValueError(
                "Benchmark not set. `benchmark_prices` should have the same number of datapoints as `prices`"
            )

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

    def _rate_conversion(self, annual_rate):
        return (annual_rate + 1) ** (1 / self.frequency) - 1

    def excess_mar(self, annual_mar=0.03, annual=True, compounding=True):
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
        _checks._check_booleans(percentage=percentage)

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

    def ewm(
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
                + self.returns.ewm(com, span, halflife, alpha, **kwargs).mean().iloc[-1]
            ) ** self.frequency - 1
        elif annual and not compounding:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **kwargs).mean().iloc[-1]
                * self.frequency
            )
        elif not annual:
            mean = (
                self.returns.ewm(com, span, halflife, alpha, **kwargs).mean().iloc[-1]
            )

        return mean[0]

    def plot_aum(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.state[self.name].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("AUM ($)")
        ax.set_title("Assets Under Management")
        if save:
            plt.savefig("aum.png", dpi=300)
        if show:
            plt.show()

    def plot_returns(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.set_title("Portfolio Returns")
        if save:
            plt.savefig("returns.png", dpi=300)
        if show:
            plt.show()

    def plot_returns_distribution(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.returns.plot.hist(ax=ax, bins=90)
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency")
        ax.set_title("Portfolio Returns Distribution")
        if save:
            plt.savefig("returns_distribution.png", dpi=300)
        if show:
            plt.show()

    def plot_cumulative_returns(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        self.cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Portfolio Cumulative Returns")
        if save:
            plt.savefig("cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_piechart(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        wp = {"linewidth": 1, "edgecolor": "black"}
        explode = tuple(repeat(0.05, len(self.tickers)))

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
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
        ax.set_title(str(self.name + " Asset Distribution"))
        if save:
            plt.savefig(str(self.name + "_pie_chart.png"), dpi=300)
        if show:
            plt.show()

    def plot_assets_cumulative_returns(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        assets_cumulative_returns = (self.assets_returns + 1).cumprod()

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
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
        annual_rfr=0.03,
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
        annual_rfr=0.03,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        capm = self.capm(
            annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        rfr = self._rate_conversion(annual_rfr)
        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
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
        annual_rfr=0.03,
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

    def excess_benchmark(
        self,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(annual=annual)
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

        tracking_error = np.std(self.returns - self.benchmark_returns.to_numpy())

        if annual:
            return tracking_error[0] * np.sqrt(self.frequency)
        else:
            return tracking_error[0]

    def information_ratio(
        self,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        excess_return = self.excess_benchmark(
            annual=annual,
            compounding=compounding,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        tracking_error = self.tracking_error(
            annual=annual,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        information_ratio = excess_return / tracking_error

        return information_ratio

    def volatility_skewness(self, annual_mar=0.03):
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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

        capm = self.capm(
            annual_rfr=annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

        capm = self.capm(
            annual_rfr=annual_rfr,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )

        if annual and compounding:
            treynor_ratio = (self.geometric_mean - annual_rfr) / capm[1]
        elif annual and not compounding:
            treynor_ratio = (self.arithmetic_mean - annual_rfr) / capm[1]
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            treynor_ratio = (self.mean - rfr) / capm[1]

        return treynor_ratio

    def hpm(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.maximum(self.returns - mar, 0), moment)
        )

        return higher_partial_moment[0]

    def lpm(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(moment=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.maximum(mar - self.returns, 0), moment)
        )

        return lower_partial_moment[0]

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
            kappa_ratio = (self.mean - mar) / np.power(
                lower_partial_moment, (1 / moment)
            )

        return kappa_ratio

    def gain_loss(self):
        higher_partial_moment = self.hpm(annual_mar=0, moment=1)
        lower_partial_moment = self.lpm(annual_mar=0, moment=1)

        gain_loss_ratio = higher_partial_moment / lower_partial_moment

        return gain_loss_ratio

    def calmar(
        self, periods=0, inverse=True, annual_rfr=0.03, annual=True, compounding=True
    ):
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

    def ulcer(self, periods=14):
        periods = _checks._check_periods(periods=periods, state=self.state)

        ulcer = pd.DataFrame(columns=["Ulcer Index"], index=self.state.index)
        period_high_close = (
            self.state[self.name]
            .rolling(periods + 1)
            .apply(lambda x: np.amax(x), raw=True)
        )
        percentage_drawdown_squared = (
            (self.state[self.name] - period_high_close) / period_high_close * 100
        ) ** 2
        squared_average = (
            percentage_drawdown_squared.rolling(periods + 1).sum() / periods
        )
        ulcer["Ulcer Index"] = np.sqrt(squared_average)

        return ulcer

    def martin(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        periods=14,
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        ulcer_index = self.ulcer(periods)

        if annual and compounding:
            martin_ratio = (self.geometric_mean - annual_rfr) / ulcer_index.iloc[-1]
        elif annual and not compounding:
            martin_ratio = (self.arithmetic_mean - annual_rfr) / ulcer_index.iloc[-1]
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            martin_ratio = (self.mean - rfr) / ulcer_index.iloc[-1]

        return martin_ratio[0]

    def plot_ulcer(
        self,
        periods=14,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        ulcer = self.ulcer(periods)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ulcer["Ulcer Index"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ulcer Index")
        ax.set_title("Portfolio Ulcer Index")
        if save:
            plt.savefig("ulcer.png", dpi=300)
        if show:
            plt.show()

    def parametric_var(self, ci=0.95, frequency=1):
        return stats.norm.ppf(1 - ci, self.mean, self.volatility) * np.sqrt(frequency)

    def historical_var(self, ci=0.95, frequency=1):
        return np.percentile(self.returns, 100 * (1 - ci)) * np.sqrt(frequency)

    def plot_parametric_var(
        self,
        ci=0.95,
        frequency=1,
        plot_z=3,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        var = self.parametric_var(ci, frequency)
        x = np.linspace(
            self.mean - plot_z * self.volatility,
            self.mean + plot_z * self.volatility,
            100,
        )
        pdf = stats.norm(self.mean, self.volatility).pdf(x)

        cutoff = (np.abs(x - var)).argmin()

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ax.plot(
            x,
            pdf,
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
        ax.set_title("Parametric VaR Plot")
        if save:
            plt.savefig("parametric_var.png", dpi=300)
        if show:
            plt.show()

    def plot_historical_var(
        self,
        ci=0.95,
        frequency=1,
        number_of_bins=100,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        var = self.historical_var(ci, frequency)
        sorted_returns = np.sort(self.returns, axis=0)
        bins = np.linspace(
            sorted_returns[0],
            sorted_returns[-1],
            number_of_bins,
        )[:, 0]

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        ax.hist(
            sorted_returns,
            bins,
            label="Historical Distribution",
        )
        ax.axvline(x=var, ymin=0, color="r", label="Value-At-Risk Cutoff")
        ax.legend()
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency of Returns")
        ax.set_title("Historical VaR Plot")
        if save:
            plt.savefig("historical_var.png", dpi=300)
        if show:
            plt.show()

    def correlation(self):
        matrix = self.assets_returns.corr().round(5)

        return matrix

    def plot_correlation(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.correlation()

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        sns.heatmap(matrix, vmin=-1, vmax=1, center=0, annot=True, ax=ax)
        ax.set_title("Correlation Matrix")
        if save:
            plt.savefig("correlation.png", dpi=300)
        if show:
            plt.show()

    def covariance(self, method="regular", annual=False, cov_kwargs={}):
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
            raise ValueError()

        if annual:
            matrix = matrix * self.frequency

        return matrix

    def plot_covariance(
        self,
        method="regular",
        annual=False,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        cov_kwargs={},
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.covariance(method, annual, **cov_kwargs)

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        sns.heatmap(matrix, vmin=0, center=0, annot=True, ax=ax)
        ax.set_title("Covariance Matrix")
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

    def omega_sharpe_ratio(self, annual_mar=0.03):
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
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        if returns is None:
            returns = self.returns

        _checks._check_omega_multiple_returns(returns)
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

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        all_values.plot(ax=ax)
        ax.set_xlabel("Minimum Acceptable Return (%)")
        ax.set_ylabel("Omega Ratio")
        ax.set_title("Omega Curve")
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

    def appraisal(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        capm = self.capm(
            annual_rfr, benchmark_prices, benchmark_weights, benchmark_name
        )

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
        **kwargs
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        _checks._check_booleans(modified=modified)

        drawdowns = self.sorted_drawdowns(largest=largest, **kwargs)
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

        return -d_ratio[0]

    def kelly_criterion(self, annual_rfr=0.03, half=True):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)
        _checks._check_booleans(half=half)

        rfr = self._rate_conversion(annual_rfr)
        excess_returns = self.returns - rfr
        kelly_criterion = excess_returns.mean() / np.var(self.returns)

        if half:
            kelly_criterion = kelly_criterion / 2

        return kelly_criterion[0]

    def modigliani(
        self,
        annual_rfr=0.03,
        annual=True,
        compounding=True,
        adjusted=False,
        probabilistic=False,
        sharpe_benchmark=0.0,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

        fama_beta = np.std(self.returns) / np.std(self.benchmark_returns.to_numpy())

        return fama_beta[0]

    def diversification(
        self,
        annual_rfr=0.03,
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
            annual_rfr=annual_rfr,
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

    def net_selectivity(self, annual_rfr=0.03, annual=True, compounding=True):
        jensen_alpha = self.jensen_alpha(annual_rfr, annual, compounding)
        diversification = self.diversification(annual_rfr, annual, compounding)

        net_selectivity = jensen_alpha - diversification

        return net_selectivity

    def downside_frequency(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        losing = self.returns[self.returns <= mar].dropna()

        downside_frequency = losing.shape[0] / self.returns.shape[0]

        return downside_frequency

    def upside_frequency(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        winning = self.returns[self.returns > mar].dropna()

        upside_frequency = winning.shape[0] / self.returns.shape[0]

        return upside_frequency

    def up_capture(
        self,
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        benchmark_prices=None,
        benchmark_weights=None,
        benchmark_name="Benchmark Portfolio",
    ):
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        set_benchmark = _checks._check_benchmark(
            slf_benchmark_prices=self.benchmark_prices,
            benchmark_prices=benchmark_prices,
            benchmark_weights=benchmark_weights,
            benchmark_name=benchmark_name,
        )
        if set_benchmark:
            self._set_benchmark(benchmark_prices, benchmark_weights, benchmark_name)

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
        _checks._check_periods(periods=periods)
        _checks._check_booleans(inverse=inverse)

        drawdowns = self.drawdowns()

        if inverse:
            mdd = -drawdowns[-periods:].min()[0]
        else:
            mdd = drawdowns[-periods:].min()[0]

        return mdd

    def average_drawdown(self, largest=0, inverse=True):
        _checks._check_nonnegints(largest=largest)
        _checks._check_booleans(inverse=inverse)

        drawdowns = self.drawdowns()
        drawdowns = drawdowns.sort_values(by=self.name, ascending=False)[-largest:]

        if inverse:
            add = -drawdowns.mean()[0]
        else:
            add = drawdowns.mean()[0]

        return add

    def sorted_drawdowns(self, largest=0, **kwargs):
        _checks._check_nonnegints(largest=largest)

        drawdowns = self.drawdowns()
        sorted_drawdowns = drawdowns.sort_values(
            by=self.name, ascending=False, **kwargs
        )[-largest:]

        return sorted_drawdowns

    def plot_drawdowns(
        self,
        style="./portfolio_analytics/portfolio_analytics_style.mplstyle",
        rcParams_update={},
        show=True,
        save=False,
        **fig_kw
    ):
        _checks._check_plot_arguments(show=show, save=save)

        drawdowns = self.drawdowns()

        plt.style.use(style)
        plt.rcParams.update(rcParams_update)
        fig, ax = plt.subplots(**fig_kw)
        drawdowns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdowns")
        ax.set_title("Portfolio Drawdowns")
        if save:
            plt.savefig("drawdowns.png", dpi=300)
        if show:
            plt.show()

    def upside_risk(self, annual_mar=0.03):
        upside_risk = np.sqrt(self.hpm(annual_mar=annual_mar, moment=2))

        return upside_risk

    def upside_potential(self, annual_mar=0.03):
        upside_potential = self.hpm(annual_mar=annual_mar, moment=1)

        return upside_potential

    def upside_variance(self, annual_mar=0.03):
        upside_variance = self.hpm(annual_mar=annual_mar, moment=2)

        return upside_variance

    def downside_risk(self, annual_mar=0.03):
        downside_risk = np.sqrt(self.lpm(annual_mar=annual_mar, moment=2))

        return downside_risk

    def downside_potential(self, annual_mar=0.03):
        downside_potential = self.lpm(annual_mar=annual_mar, moment=1)

        return downside_potential

    def downside_variance(self, annual_mar=0.03):
        downside_variance = self.lpm(annual_mar=annual_mar, moment=2)

        return downside_variance
