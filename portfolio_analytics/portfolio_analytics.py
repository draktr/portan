import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import lilliefors
from itertools import repeat
from portfolio_analytics import _checks


class PortfolioAnalytics:
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):
        self.prices = prices
        self.assets_returns = self.prices.pct_change().drop(self.prices.index[0])
        self.tickers = self.prices.columns
        self.weights = weights
        self.assets_info = pdr.get_quote_yahoo(self.tickers)
        self.assets_names = self.assets_info["longName"]
        self.portfolio_name = portfolio_name
        self.initial_aum = initial_aum
        self.frequency = frequency

        # funds allocated to each asset
        self.allocation_funds = np.multiply(self.initial_aum, self.weights)
        self.allocation_funds = pd.Series(self.allocation_funds, index=self.tickers)

        # number of assets bought at t0
        self.allocation_assets = np.divide(self.allocation_funds, self.prices.iloc[0].T)
        self.allocation_assets = pd.Series(self.allocation_assets, index=self.tickers)

        # absolute (dollar) value of each asset in portfolio (i.e. state of the portfolio, not rebalanced)
        self.portfolio_state = pd.DataFrame(
            np.multiply(self.prices, self.allocation_assets),
            index=self.prices.index,
            columns=self.tickers,
        )
        self.portfolio_state["Whole Portfolio"] = self.portfolio_state.sum(axis=1)

        self.portfolio_returns = np.dot(self.assets_returns.to_numpy(), self.weights)
        self.portfolio_returns = pd.Series(
            self.portfolio_returns,
            index=self.assets_returns.index,
            name=self.portfolio_name,
        )

        self.portfolio_cumulative_returns = (self.portfolio_returns + 1).cumprod()

        self.mean = self.portfolio_returns.mean()
        self.arithmetic_mean = self.mean * self.frequency
        self.geometric_mean = (1 + self.portfolio_returns).prod() ** (
            self.frequency / self.portfolio_returns.shape[0]
        ) - 1

        self.volatility = self.portfolio_returns.std()
        self.annual_volatility = self.volatility * np.sqrt(self.frequency)

    def _rate_conversion(self, annual_rate):
        return (annual_rate + 1) ** (1 / self.frequency) - 1


class ExploratoryQuantitativeAnalytics(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):
        super().__init__(prices, weights, portfolio_name, initial_aum, frequency)

    def excess_returns(self, annual_mar=0.03):
        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)
        excess_returns = self.portfolio_returns - mar

        return excess_returns

    def net_return(self, percentage=False):
        _checks._check_percentage(percentage=percentage)

        final_aum = self.final_aum()

        if not percentage:
            net_return = final_aum - self.initial_aum
        else:
            net_return = (final_aum - self.initial_aum) / self.initial_aum

        return net_return

    def min_aum(self):
        min_aum = self.portfolio_state["Whole Portfolio"].min()

        return min_aum

    def max_aum(self):
        max_aum = self.portfolio_state["Whole Portfolio"].max()

        return max_aum

    def mean_aum(self):
        mean_aum = self.portfolio_state["Whole Portfolio"].mean()

        return mean_aum

    def final_aum(self):
        final_aum = self.allocation_assets * self.assets_info["regularMarketPrice"]

        return final_aum

    def distribution_test(self, test="dagostino-pearson", distribution="norm"):
        if test == "dagostino-pearson":
            result = stats.normaltest(self.portfolio_returns)
        elif test == "kolomogorov-smirnov":
            result = stats.kstest(self.portfolio_returns, distribution)
        elif test == "lilliefors":
            result = lilliefors(self.portfolio_returns)
        elif test == "shapiro-wilk":
            result = stats.shapiro(self.portfolio_returns)
        elif test == "jarque-barre":
            result = stats.jarque_bera(self.portfolio_returns)
        elif test == "anderson-darling":
            result = stats.anderson(self.portfolio_returns, distribution)
        else:
            raise ValueError("Statistical test is unavailable.")

        return result


class ExploratoryVisualAnalytics(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):
        super().__init__(prices, weights, portfolio_name, initial_aum, frequency)

    def plot_aum(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_state["Whole Portfolio"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("AUM ($)")
        ax.set_title("Assets Under Management")
        if save:
            plt.savefig("aum.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_returns(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.set_title("Portfolio Returns")
        if save:
            plt.savefig("portfolio_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_returns_distribution(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_returns.plot.hist(bins=90)
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency")
        ax.set_title("Portfolio Returns Distribution")
        if save:
            plt.savefig("portfolio_returns_distribution.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_cumulative_returns(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Portfolio Cumulative Returns")
        if save:
            plt.savefig("portfolio_cumulative_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_piechart(self, show=True, save=False):
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
        ax.set_title(str(self.portfolio_name + " Asset Distribution"))
        if save:
            plt.savefig(str(self.portfolio_name + "_pie_chart.png"), dpi=300)
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


class MPT(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        benchmark_prices,
        benchmark_weights,
        portfolio_name="Investment Portfolio",
        benchmark_name="Benchmark",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

        self.benchmark_name = benchmark_name

        self.benchmark_assets_returns = benchmark_prices.pct_change().drop(
            benchmark_prices.index[0]
        )

        self.benchmark_returns = np.dot(
            self.benchmark_assets_returns.to_numpy(), benchmark_weights
        )
        self.benchmark_returns = np.delete(self.benchmark_returns, [0], axis=0)
        self.benchmark_returns = pd.DataFrame(
            self.benchmark_returns,
            index=self.benchmark_assets_returns.index,
            columns=[benchmark_name],
        )

        self.benchmark_geometric_mean = (1 + self.benchmark_returns).prod() ** (
            self.frequency / self.benchmark_returns.shape[0]
        ) - 1
        self.benchmark_arithmetic_mean = self.benchmark_returns.mean() * self.frequency
        self.benchmark_mean = self.benchmark_assets_returns.mean()

    def capm(self, annual_rfr=0.02):
        _checks._check_rate_arguments(annual_rfr=annual_rfr)

        rfr = self._rate_conversion(annual_rfr)

        excess_portfolio_returns = self.portfolio_returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        alpha = model.intercept_
        beta = model.coef_[0]
        r_squared = model.score(excess_benchmark_returns, excess_portfolio_returns)

        return (
            alpha,
            beta,
            r_squared,
            excess_portfolio_returns,
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
        tracking_error = np.std(self.portfolio_returns - self.benchmark_returns, ddof=1)

        return tracking_error


class PMPT(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        benchmark_prices,
        benchmark_weights,
        portfolio_name="Investment Portfolio",
        benchmark_name="Benchmark",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

        self.benchmark_name = benchmark_name

        self.benchmark_assets_returns = benchmark_prices.pct_change().drop(
            benchmark_prices.index[0]
        )

        self.benchmark_returns = np.dot(
            self.benchmark_assets_returns.to_numpy(), benchmark_weights
        )
        self.benchmark_returns = np.delete(self.benchmark_returns, [0], axis=0)
        self.benchmark_returns = pd.DataFrame(
            self.benchmark_returns,
            index=self.benchmark_assets_returns.index,
            columns=[benchmark_name],
        )

        self.benchmark_geometric_mean = (1 + self.benchmark_returns).prod() ** (
            self.frequency / self.benchmark_returns.shape[0]
        ) - 1
        self.benchmark_arithmetic_mean = self.benchmark_returns.mean() * self.frequency
        self.benchmark_mean = self.benchmark_assets_returns.mean()

    def upside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        positive_portfolio_returns = self.portfolio_returns - mar
        positive_portfolio_returns = positive_portfolio_returns[
            positive_portfolio_returns > 0
        ]
        if annual:
            upside_volatility = (
                np.std(positive_portfolio_returns, ddof=1) * self.frequency
            )
        else:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1)

        return upside_volatility

    def downside_volatility(self, annual_mar=0.03, annual=True):
        _checks._check_rate_arguments(annual_mar=annual_mar, annual=annual)

        mar = self._rate_conversion(annual_mar)

        negative_portfolio_returns = self.portfolio_returns - mar
        negative_portfolio_returns = negative_portfolio_returns[
            negative_portfolio_returns < 0
        ]
        if annual:
            downside_volatility = (
                np.std(negative_portfolio_returns, ddof=1) * self.frequency
            )
        else:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1)

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
            self.portfolio_returns
            - 3 * portfolio_downside_volatility * benchmark_downside_volatility
        )

        return omega_excess_return

    def upside_potential_ratio(self, annual_mar=0.03, annual=True):
        mar = self._rate_conversion(annual_mar)

        downside_volatility = self.downside_volatility(annual_mar, annual)
        upside = self.portfolio_returns - mar
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

        negative_portfolio_returns = self.portfolio_returns - mar
        negative_portfolio_returns = negative_portfolio_returns[
            negative_portfolio_returns < 0
        ]

        model = LinearRegression().fit(
            negative_benchmark_returns, negative_portfolio_returns
        )
        downside_alpha = model.intercept_
        downside_beta = model.coef_[0]
        downside_r_squared = model.score(
            negative_benchmark_returns, negative_portfolio_returns
        )

        return (
            downside_beta,
            downside_alpha,
            downside_r_squared,
            negative_portfolio_returns,
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
        wealth_index = 1000 * (1 + self.portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return drawdowns

    def maximum_drawdown(self, period=1000, percentage=False):
        period = _checks._check_period(
            period=period, portfolio_state=self.portfolio_state
        )
        _checks._check_percentage(percentage=percentage)

        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

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

        excess_portfolio_returns = self.portfolio_returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
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

        excess_portfolio_returns = self.portfolio_returns - rfr
        excess_benchmark_returns = self.benchmark_returns - rfr

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        beta = model.coef_[0]

        if annual and compounding:
            treynor_ratio = 100 * (self.geometric_mean - annual_rfr) / beta
        elif annual and not compounding:
            treynor_ratio = 100 * (self.arithmetic_mean - annual_rfr) / beta
        elif not annual:
            treynor_ratio = 100 * (self.mean - rfr) / beta

        return treynor_ratio

    def higher_partial_moment(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(argument=moment)

        mar = self._rate_conversion(annual_mar)

        days = self.portfolio_returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.max(self.portfolio_returns - mar, 0), moment)
        )

        return higher_partial_moment

    def lower_partial_moment(self, annual_mar=0.03, moment=3):
        _checks._check_rate_arguments(annual_mar=annual_mar)
        _checks._check_posints(argument=moment)

        mar = self._rate_conversion(annual_mar)
        days = self.portfolio_returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.max(mar - self.portfolio_returns, 0), moment)
        )

        return lower_partial_moment

    def kappa(self, annual_mar=0.03, moment=3, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_mar=annual_mar, annual=annual, compounding=compounding
        )
        _checks._check_posints(argument=moment)

        lower_partial_moment = self.lower_partial_moment(annual_mar, moment)

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
        hpm = self.higher_partial_moment(annual_mar, moment)
        lpm = self.lower_partial_moment(annual_mar, moment)

        gain_loss_ratio = hpm / lpm

        return gain_loss_ratio

    def calmar(self, period=1000, annual_rfr=0.02, annual=True, compounding=True):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )

        maximum_drawdown = self.maximum_drawdown(period=period, percentage=True)

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
        _checks._check_posints(argument=drawdowns)

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


class Ulcer(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

    def ulcer(self, period=14, start=1):
        period = _checks._check_period(
            period=period, portfolio_state=self.portfolio_state
        )
        _checks._check_posints(argument=start)

        close = np.empty(period)
        percentage_drawdown = np.empty(period)

        if start == 1:
            period_high = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        else:
            period_high = np.max(
                self.portfolio_state.iloc[-period - start + 1 : -start + 1][
                    "Whole Portfolio"
                ]
            )

        for i in range(period):
            close[i] = self.portfolio_state.iloc[-i - start + 1]["Whole Portfolio"]
            percentage_drawdown[i] = 100 * ((close[i] - period_high)) / period_high

        ulcer_index = np.sqrt(np.mean(np.square(percentage_drawdown)))

        return ulcer_index

    def martin(
        self,
        annual_rfr=0.02,
        annual=True,
        compounding=True,
        period=14,
    ):
        _checks._check_rate_arguments(
            annual_rfr=annual_rfr, annual=annual, compounding=compounding
        )
        period = _checks._check_period(
            period=period, portfolio_state=self.portfolio_state
        )

        ulcer_index = self.ulcer(period)

        if annual and compounding:
            martin_ratio = 100 * (self.geometric_mean - annual_rfr) / ulcer_index
        elif annual and not compounding:
            martin_ratio = 100 * (self.arithmetic_mean - annual_rfr) / ulcer_index
        elif not annual:
            rfr = self._rate_conversion(annual_rfr)
            martin_ratio = 100 * (self.mean - rfr) / ulcer_index

        return martin_ratio

    def ulcer_series(self, period=14):
        period = _checks._check_period(
            period=period, portfolio_state=self.portfolio_state
        )

        ulcer_series = pd.DataFrame(
            columns=["Ulcer Index"], index=self.portfolio_state.index
        )
        for i in range(self.portfolio_state.shape[0] - period):
            ulcer_series.iloc[-i]["Ulcer Index"] = self.ulcer(period, start=i)

        return ulcer_series

    def plot_ulcer(self, period=14, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        ulcer_series = self.ulcer_series(period)

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


class ValueAtRisk(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

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
        returns_below_value = self.portfolio_returns[self.portfolio_returns < value]
        var = returns_below_value.shape[0] / self.portfolio_returns.shape[0]

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

        sorted_portfolio_returns = np.sort(self.portfolio_returns)
        bins = np.linspace(
            sorted_portfolio_returns[0],
            sorted_portfolio_returns[-1] + 1,
            number_of_bins,
        )

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.hist(
            sorted_portfolio_returns,
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


class Matrices(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:
        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

    def correlation_matrix(self):
        matrix = self.portfolio_returns.corr().round(5)

        return matrix

    def plot_correlation_matrix(self, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.correlation_matrix()

        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
        if save:
            plt.savefig("correlation_matrix.png", dpi=300)
        if show:
            plt.show()

    def covariance_matrix(self, annual=False):
        if annual:
            matrix = self.portfolio_returns.cov().round(5) * self.frequency
        else:
            matrix = self.portfolio_returns.cov().round(5)

        return matrix

    def plot_covariance_matrix(self, annual=False, show=True, save=False):
        _checks._check_plot_arguments(show=show, save=save)

        matrix = self.covariance_matrix(annual)

        sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
        if save:
            plt.savefig("covariance_matrix.png", dpi=300)
        if show:
            plt.show()


class OmegaAnalysis(PortfolioAnalytics):
    def __init__(
        self,
        prices,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
        annual_mar_lower_bound=0,
        annual_mar_upper_bound=0.2,
    ):
        """
        Initiates the object

        :param prices: Prices data for all assets in portfolio.
        :type prices: pd.DataFrame
        :param weights: Asset weights in portfolio.
        :type weights: list or np.ndarray
        :param portfolio_name: Name of the innvestment portfolio being analysed., defaults to "Investment Portfolio"
        :type portfolio_name: str, optional
        :param initial_aum: Initial Assets Under Management, defaults to 10000
        :type initial_aum: int, optional
        :param frequency: Number of values in the data in one calendar year, defaults to 252
        :type frequency: int, optional
        :param annual_mar_lower_bound: Annual Minimum Acceptable Return (MAR) lower bound for the Omega Curve., defaults to 0
        :type annual_mar_lower_bound: int or float, optional
        :param annual_mar_upper_bound: Annual Minimum Acceptable Return (MAR) upper bound for the Omega Curve., defaults to 0.2
        :type annual_mar_upper_bound: int or float, optional
        """

        super.__init__(prices, weights, portfolio_name, initial_aum, frequency)

        self.mar_array = np.linspace(
            annual_mar_lower_bound,
            annual_mar_upper_bound,
            round(100 * (annual_mar_upper_bound - annual_mar_lower_bound)),
        )

    def omega_ratio(self, annual_mar=0.03):
        """
        Calculates the Omega Ratio of the portfolio.

        :param annual_mar: Annual Minimum Acceptable Return (MAR)., defaults to 0.03
        :type annual_mar: float, optional
        :return: Omega Ratio of the portfolio
        :rtype: pd.DataFrame
        """

        _checks._check_rate_arguments(annual_mar=annual_mar)

        mar = self._rate_conversion(annual_mar)

        excess_returns = self.portfolio_returns - mar
        winning = excess_returns[excess_returns > 0].sum()
        losing = -(excess_returns[excess_returns <= 0].sum())

        omega = winning / losing

        return omega

    def plot_omega_curve(self, returns=None, show=True, save=False):
        """
        Plots and/or saves Omega Curve(s) of the portfolio(s)

        :param show: Show the plot upon the execution of the code., defaults to True
        :type show: bool, optional
        :param save: Save the plot on storage., defaults to False
        :type save: bool, optional
        """

        if returns is None:
            returns = self.portfolio_returns

        _checks._check_multiple_returns(returns=returns)
        _checks._check_plot_arguments(show=show, save=save)

        all_values = pd.DataFrame(columns=returns.columns)

        for portfolio in returns.columns:
            omega_values = list()
            for mar in self.mar_array:
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
