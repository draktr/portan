import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn import covariance
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import lilliefors
from itertools import repeat
from portfolio_analytics import _checks


class PortfolioAnalytics:
    def __init__(
        self,
        prices,
        weights,
        benchmark_prices,
        benchmark_weights,
        name="Investment Portfolio",
        benchmark_name="Benchmark Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        _checks._check_init(
            prices=prices,
            weights=weights,
            name=name,
            initial_aum=initial_aum,
            frequency=frequency,
        )

        self.prices = prices
        self.assets_returns = self.prices.pct_change().drop(self.prices.index[0])
        self.tickers = self.prices.columns
        self.weights = weights
        self.assets_info = pdr.get_quote_yahoo(self.tickers)
        self.assets_names = self.assets_info["longName"]
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
        self.state["Whole Portfolio"] = self.state.sum(axis=1)

        self.returns = pd.Series(
            np.dot(self.assets_returns.to_numpy(), self.weights),
            index=self.assets_returns.index,
            name=self.name,
        )

        self.cumulative_returns = (self.returns + 1).cumprod()

        self.mean = self.returns.mean()
        self.arithmetic_mean = self.mean * self.frequency
        self.geometric_mean = (1 + self.returns).prod() ** (
            self.frequency / self.returns.shape[0]
        ) - 1

        self.volatility = self.returns.std()
        self.annual_volatility = self.volatility * np.sqrt(self.frequency)

        self.min_aum = self.state["Whole Portfolio"].min()
        self.max_aum = self.state["Whole Portfolio"].max()
        self.mean_aum = self.state["Whole Portfolio"].mean()
        self.final_aum = np.sum(self.allocation_assets * self.state.iloc[-1, 0:-1])

        self.benchmark_assets_returns = self.benchmark_prices.pct_change().drop(
            self.benchmark_prices.index[0]
        )

        self.benchmark_returns = pd.DataFrame(
            np.dot(self.benchmark_assets_returns.to_numpy(), self.benchmark_weights),
            index=self.benchmark_assets_returns.index,
            columns=[self.benchmark_name],
        )

        self.benchmark_mean = self.benchmark_returns.mean()
        self.benchmark_arithmetic_mean = self.benchmark_returns.mean() * self.frequency
        self.benchmark_geometric_mean = (1 + self.benchmark_returns).prod() ** (
            self.frequency / self.benchmark_returns.shape[0]
        ) - 1

    def _rate_conversion(self, annual_rate):
        return (annual_rate + 1) ** (1 / self.frequency) - 1

    def excess_returns(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        excess_returns = self.returns - mar

        return excess_returns

    def net_return(self, percentage=False):
        _checks._check_percentage(percentage=percentage)

        final_aum = self.final_aum()

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

    def ewm_return(self, span, annual=True, compounding=True):
        _checks._check_rate_arguments(annual=annual, compounding=compounding)

        if annual and compounding:
            mean = (
                1 + self.returns.ewm(span=span).mean().iloc[-1]
            ) ** self.frequency - 1
        elif annual and not compounding:
            mean = self.returns.ewm(span=span).mean().iloc[-1] * self.frequency
        elif not annual:
            mean = self.returns.ewm(span=span).mean().iloc[-1]

        return mean

    def plot_aum(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.state["Whole Portfolio"].plot(ax=ax)
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
        self.returns.plot.hist(bins=90)
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

    def capm_return(self, annual_rfr=0.02, annual=True, compounding=True):
        capm = self.capm(annual_rfr=annual_rfr)

        if annual and compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_geometric_mean - annual_rfr)
        elif annual and not compounding:
            mean = annual_rfr + capm[1] * (self.benchmark_arithmetic_mean - annual_rfr)
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            mean = rfr + capm[1] * (self.benchmark_mean - rfr)

        return mean

    def capm(self, annual_rfr=0.02):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)

        rfr = self._rate_conversion(annual_rfr)

        excess_returns = self.returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(excess_benchmark_returns, excess_returns)
        alpha = model.intercept_
        beta = model.coef_[0]
        r_squared = model.score(excess_benchmark_returns, excess_returns)

        return (
            alpha,
            beta,
            r_squared,
            excess_returns,
            excess_benchmark_returns,
        )

    def plot_capm(self, annual_rfr=0.02, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        capm = self.capm(annual_rfr)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(capm[4], capm[3], color="b")
        ax.plot(capm[4], capm[0] + capm[1] * capm[4], color="r")
        empty_patch = mpatches.Patch(color="none", visible=False)
        ax.legend(
            handles=[empty_patch, empty_patch],
            labels=[
                r"$\alpha$" + " = " + str(np.round(capm[0], 3)),
                r"$\beta$" + " = " + str(np.round(capm[1], 3)),
            ],
        )
        ax.set_xlabel("Benchmark Excess Returns")
        ax.set_ylabel("Portfolio Excess Returns")
        ax.set_title("Portfolio Excess Returns Against Benchmark (CAPM)")
        if save:
            plt.savefig("capm.png", dpi=300)
        if show:
            plt.show()

    def sharpe(self, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

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

        return sharpe_ratio

    def tracking_error(self):
        tracking_error = np.std(self.returns - self.benchmark_returns, ddof=1)

        return tracking_error

    def upside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        positive_returns = self.returns - mar
        positive_returns = positive_returns[positive_returns > 0]
        if annual:
            upside_volatility = np.std(positive_returns, ddof=1) * self.frequency
        else:
            upside_volatility = np.std(positive_returns, ddof=1)

        return upside_volatility

    def downside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        negative_returns = self.returns - mar
        negative_returns = negative_returns[negative_returns < 0]
        if annual:
            downside_volatility = np.std(negative_returns, ddof=1) * self.frequency
        else:
            downside_volatility = np.std(negative_returns, ddof=1)

        return downside_volatility

    def volatility_skew(self, annual_mar=0.03, annual=True):
        upside = self.upside_volatility(annual_mar, annual)
        downside = self.downside_volatility(annual_mar, annual)
        skew = upside / downside

        return skew

    def omega_excess_return(self, annual_mar=0.03, annual=True):
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

        omega_excess_return = (
            self.returns
            - 3 * portfolio_downside_volatility * benchmark_downside_volatility
        )

        return omega_excess_return

    def upside_potential(self, annual_mar=0.03, annual=True):
        mar = self._rate_conversion(annual_mar)

        downside_volatility = self.downside_volatility(annual_mar, annual)
        upside = self.returns - mar
        upside = upside[upside > 0].sum()
        upside_potential_ratio = upside / downside_volatility

        return upside_potential_ratio

    def downside_capm(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)

        negative_benchmark_returns = self.benchmark_returns - mar
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]

        negative_returns = self.returns - mar
        negative_returns = negative_returns[negative_returns < 0]

        model = LinearRegression().fit(negative_benchmark_returns, negative_returns)
        downside_alpha = model.intercept_
        downside_beta = model.coef_[0]
        downside_r_squared = model.score(negative_benchmark_returns, negative_returns)

        return (
            downside_beta,
            downside_alpha,
            downside_r_squared,
            negative_returns,
            negative_benchmark_returns,
        )

    def downside_volatility_ratio(self, annual_mar=0.03, annual=True):
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

        return downside_volatility_ratio

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

    def drawdowns(self):
        wealth_index = 1000 * (1 + self.returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return drawdowns

    def maximum_drawdown(self, periods=1000, percentage=False):
        periods = _checks._check_periods(periods=periods, state=self.state)
        _checks._check_percentage(percentage=percentage)

        peak = np.max(self.state.iloc[-periods:]["Whole Portfolio"])
        peak_index = self.state["Whole Portfolio"].idxmax()
        peak_index_int = self.state.index.get_loc(peak_index)
        trough = np.min(self.state.iloc[-peak_index_int:]["Whole Portfolio"])

        if not percentage:
            maximum_drawdown = trough - peak
        else:
            maximum_drawdown = (trough - peak) / peak

        return maximum_drawdown

    def jensen_alpha(self, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
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

        return jensen_alpha

    def treynor(self, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
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

        return treynor_ratio

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

        maximum_drawdown = self.maximum_drawdown(periods=periods, percentage=True)

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
            periods_high = np.max(self.state.iloc[-periods:]["Whole Portfolio"])
        else:
            periods_high = np.max(
                self.state.iloc[-periods - start + 1 : -start + 1]["Whole Portfolio"]
            )

        for i in range(periods):
            close[i] = self.state.iloc[-i - start + 1]["Whole Portfolio"]
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
            label="Analytical (Theoretical) Distribution of Portfolio Returns",
        )
        ax.fill_between(
            x[0:cutoff], pdf[0:cutoff], facecolor="r", label="Analytical VaR"
        )
        ax.legend()
        ax.set_xlabel("Returns")
        ax.set_ylabel("Density of Returns")
        ax.set_title(
            "Analytical (Theoretical,"
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
        )

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.hist(
            sorted_returns,
            bins,
            label="Historical Distribution of Portfolio Returns",
        )
        ax.axvline(x=value, ymin=0, color="r", label="Historical VaR Cutoff")
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

    def omega_ratio(self, annual_mar=0.03):
        """
        Calculates the Omega ratio of the portfolio.

        :param annual_mar: Annual Minimum Acceptable Return (MAR)., defaults to 0.03
        :type annual_mar: float, optional
        :return: Omega ratio of the portfolio
        :rtype: pd.DataFrame
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)

        excess_returns = self.returns - mar
        winning = excess_returns[excess_returns > 0].sum()
        losing = -(excess_returns[excess_returns <= 0].sum())

        omega = winning / losing

        return omega

    def plot_omega_curve(
        self,
        returns=None,
        annual_mar_lower_bound=0,
        annual_mar_upper_bound=0.2,
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
            ylim=(0, 1.5),
        )
        if save:
            plt.savefig("omega_curves.png", dpi=300)
        if show:
            plt.show()
