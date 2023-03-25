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
import warnings
import inspect


class PortfolioAnalytics:
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):

        self.prices = data["Adj Close"]
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
        self.self.portfolio_state = np.multiply(self.prices, self.allocation_assets)
        self.self.portfolio_state["Whole Portfolio"] = self.self.portfolio_state.sum(
            axis=1
        )

        self.portfolio_returns = np.dot(self.assets_returns.to_numpy(), self.weights)
        self.portfolio_returns = pd.Series(
            self.portfolio_returns,
            index=self.assets_returns.index,
            name=self.portfolio_name,
        )

        self.portfolio_cumulative_returns = (self.portfolio_returns + 1).cumprod()

        self.daily_mean = self.portfolio_returns.mean()
        self.arithmetic_mean = self.daily_mean * self.frequency
        self.geometric_mean = (1 + self.portfolio_returns).prod() ** (
            self.frequency / self.portfolio_returns.shape[0]
        ) - 1

        self.daily_volatility = self.portfolio_returns.std()
        self.volatility = self.daily_volatility * np.sqrt(self.frequency)

        self.analytics = {}

    def save_executed(self):

        analytics = pd.DataFrame(
            list(self.analytics.values()), index=self.analytics.keys()
        )
        analytics.transpose().to_csv("analytics.csv")

    def save_listed(self, methods):

        analytics = {}
        for method in methods:
            analytics.update(method=getattr(self, method)())
        analytics = pd.DataFrame(list(analytics.values()), index=analytics.keys())
        analytics.transpose().to_csv("analytics.csv")

    def _rate_conversion(self, annual_rate):

        return (annual_rate + 1) ** (1 / self.frequency) - 1


class ExploratoryQuantitativeAnalytics(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):

        super().__init__(data, weights, portfolio_name, initial_aum, frequency)

    def excess_returns(self, mar=0.03):

        mar_daily = (mar + 1) ** (1 / 252) - 1
        excess_returns = self.portfolio_returns - mar_daily

        self.analytics.update({str(inspect.stack()[0][3]): excess_returns})

        return excess_returns

    def net_return(self, percentage=False):

        final_aum = self.final_aum()

        if not percentage:
            net_return = final_aum - self.initial_aum
        elif percentage:
            net_return = (final_aum - self.initial_aum) / self.initial_aum
        else:
            raise ValueError("Argument 'percentage' has to be boolean.")

        self.analytics.update({str(inspect.stack()[0][3]): net_return})

        return net_return

    def min_aum(self):

        min_aum = self.self.portfolio_state["Whole Portfolio"].min()

        self.analytics.update({str(inspect.stack()[0][3]): min_aum})

        return min_aum

    def max_aum(self):

        max_aum = self.self.portfolio_state["Whole Portfolio"].max()

        self.analytics.update({str(inspect.stack()[0][3]): max_aum})

        return max_aum

    def mean_aum(self):

        mean_aum = self.self.portfolio_state["Whole Portfolio"].mean()

        self.analytics.update({str(inspect.stack()[0][3]): mean_aum})

        return mean_aum

    def final_aum(self):

        final_aum = self.allocation_assets * self.assets_info["regularMarketPrice"]

        setattr(self, self.analytics, self.analytics.update({"final_aum": final_aum}))

        return final_aum

    def distribution_test(self, test="dagostino-pearson", distribution="norm"):

        if test == "kolomogorov-smirnov":
            result = stats.kstest(self.portfolio_returns, distribution)
        elif test == "lilliefors":
            result = lilliefors(self.portfolio_returns)
        elif test == "shapiro-wilk":
            result = stats.shapiro(self.portfolio_returns)
        elif test == "jarque-barre":
            result = stats.jarque_bera(self.portfolio_returns)
        elif test == "dagostino-pearson":
            result = stats.normaltest(self.portfolio_returns)
        elif test == "anderson-darling":
            result = stats.anderson(self.portfolio_returns, distribution)
        else:
            raise ValueError("Statistical test is unavailable.")

        self.analytics.update({str(inspect.stack()[0][3]): result})

        return result


class ExploratoryVisualAnalytics(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ):

        super().__init__(data, weights, portfolio_name, initial_aum, frequency)

    def plot_aum(self, show=True, save=False):

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

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Returns")
        ax.set_title("Portfolio Daily Returns")
        if save:
            plt.savefig("portfolio_returns.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_returns_distribution(self, show=True, save=False):

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.portfolio_returns.plot.hist(bins=90)
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Frequency")
        ax.set_title("Portfolio Returns Distribution")
        if save:
            plt.savefig("portfolio_returns_distribution.png", dpi=300)
        if show:
            plt.show()

    def plot_portfolio_cumulative_returns(self, show=True, save=False):

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

    def plot_portfolio_piechart(
        self,
        weights,
        show=True,
        save=False,
    ):

        allocation_funds = np.multiply(self.initial_aum, weights)
        wp = {"linewidth": 1, "edgecolor": "black"}
        explode = tuple(repeat(0.05, len(self.tickers)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        pie = ax.pie(
            allocation_funds,
            autopct=lambda pct: self._ap(pct, allocation_funds),
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
        data,
        weights,
        benchmark_data,
        benchmark_weights,
        portfolio_name="Investment Portfolio",
        benchmark_name="Benchmark",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

        self.benchmark_name = benchmark_name

        self.benchmark_assets_returns = (
            benchmark_data["Adj Close"].pct_change().drop(benchmark_data.index[0])
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
        self.benchmark_daily_mean = self.benchmark_assets_returns.mean()

    def capm(self, rfr=0.02):

        rfr_daily = (rfr + 1) ** (1 / 252) - 1

        excess_portfolio_returns = self.portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

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

    def plot_capm(self, rfr=0.02, show=True, save=False):

        capm = self.capm(rfr)

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

    def sharpe(self, daily=False, compounding=True, rfr=0.02):

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            sharpe_ratio = 100 * (self.geometric_mean - rfr) / self.volatility
        if not daily and not compounding:
            sharpe_ratio = 100 * (self.arithmetic_mean - rfr) / self.volatility
        elif daily:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sharpe_ratio = 100 * (self.daily_mean - rfr_daily) / self.daily_volatility

        return sharpe_ratio

    def tracking_error(self):

        tracking_error = np.std(self.portfolio_returns - self.benchmark_returns, ddof=1)

        return tracking_error


class PMPT(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        benchmark_data,
        benchmark_weights,
        portfolio_name="Investment Portfolio",
        benchmark_name="Benchmark",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

        self.benchmark_name = benchmark_name

        self.benchmark_assets_returns = (
            benchmark_data["Adj Close"].pct_change().drop(benchmark_data.index[0])
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
        self.benchmark_daily_mean = self.benchmark_assets_returns.mean()

    def upside_volatility(self, mar=0.03, daily=False):

        mar_daily = self._rate_conversion(mar)

        positive_portfolio_returns = self.portfolio_returns - mar_daily
        positive_portfolio_returns = positive_portfolio_returns[
            positive_portfolio_returns > 0
        ]
        if not daily:
            upside_volatility = (
                np.std(positive_portfolio_returns, ddof=1) * self.frequency
            )
        else:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1)

        return upside_volatility

    def downside_volatility(self, mar=0.03, daily=False):

        mar_daily = self._rate_conversion(mar)

        negative_portfolio_returns = self.portfolio_returns - mar_daily
        negative_portfolio_returns = negative_portfolio_returns[
            negative_portfolio_returns < 0
        ]
        if not daily:
            downside_volatility = (
                np.std(negative_portfolio_returns, ddof=1) * self.frequency
            )
        else:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1)

        return downside_volatility

    def volatility_skew(self, mar=0.03, daily=False):

        upside = self.upside_volatility(mar, daily)
        downside = self.downside_volatility(mar, daily)
        skew = upside / downside

        return skew

    def omega_excess_return(self, mar=0.03, daily=False):

        portfolio_downside_volatility = self.downside_volatility(mar, daily)

        mar_daily = self._rate_conversion(mar)

        negative_benchmark_returns = self.benchmark_returns - mar_daily
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]
        if not daily:
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

    def upside_potential_ratio(self, mar=0.03, daily=False):

        mar_daily = self._rate_conversion(mar)

        downside_volatility = self.downside_volatility(mar, daily)
        upside = self.portfolio_returns - mar_daily
        upside = upside[upside > 0].sum()
        upside_potential_ratio = upside / downside_volatility

        return upside_potential_ratio

    def downside_capm(self, mar=0.03):

        mar_daily = self._rate_conversion(mar)

        negative_benchmark_returns = self.benchmark_returns - mar_daily
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]

        negative_portfolio_returns = self.portfolio_returns - mar_daily
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

    def downside_volatility_ratio(self, mar=0.03, daily=False):

        portfolio_downside_volatility = self.downside_volatility(mar, daily)

        mar_daily = self._rate_conversion(mar)

        negative_benchmark_returns = self.benchmark_returns - mar_daily
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]
        if not daily:
            benchmark_downside_volatility = (
                np.std(negative_benchmark_returns, ddof=1) * self.frequency
            )
        else:
            benchmark_downside_volatility = np.std(negative_benchmark_returns, ddof=1)

        downside_volatility_ratio = (
            portfolio_downside_volatility / benchmark_downside_volatility
        )

        return downside_volatility_ratio

    def sortino(self, mar=0.03, rfr=0.02, daily=False, compounding=True):

        downside_volatility = self.downside_volatility(mar, daily)

        if daily and compounding:
            raise ValueError(
                "Mean returns cannot be compounded if daily."
            )  # TODO: think about these
        elif not daily and compounding:
            sortino_ratio = 100 * (self.geometric_mean - rfr) / downside_volatility
        elif not daily and not compounding:
            sortino_ratio = 100 * (self.arithmetic_mean - rfr) / downside_volatility
        elif daily:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sortino_ratio = 100 * (self.daily_mean - rfr_daily) / downside_volatility

        return sortino_ratio

    def drawdowns(self):

        wealth_index = 1000 * (1 + self.portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return drawdowns

    def maximum_drawdown(self, period=1000):

        if period >= self.portfolio_state.shape[0]:
            period = self.portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown = trough - peak

        return maximum_drawdown

    def maximum_drawdown_percentage(self, period=1000):

        if period > self.portfolio_state.shape[0]:
            period = self.portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown_ratio = (trough - peak) / peak

        return maximum_drawdown_ratio

    def jensen_alpha(self, rfr=0.02, daily=False, compounding=True):

        rfr_daily = self._rate_conversion(rfr)

        excess_portfolio_returns = self.portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        beta = model.coef_[0]

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            jensen_alpha = (
                self.geometric_mean - rfr - beta * (self.benchmark_geometric_mean - rfr)
            )
        if not daily and not compounding:
            jensen_alpha = (
                self.arithmetic_mean
                - rfr
                - beta * (self.benchmark_arithmetic_mean - rfr)
            )
        elif daily:
            jensen_alpha = (
                self.daily_mean
                - rfr_daily
                - beta * (self.benchmark_daily_mean - rfr_daily)
            )

        return jensen_alpha

    def treynor(self, rfr=0.02, daily=False, compounding=True):

        rfr_daily = self._rate_conversion(rfr)

        excess_portfolio_returns = self.portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        beta = model.coef_[0]

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            treynor_ratio = 100 * (self.geometric_mean - rfr) / beta
        if not daily and not compounding:
            treynor_ratio = 100 * (self.arithmetic_mean - rfr) / beta
        elif daily:
            treynor_ratio = 100 * (self.daily_mean - rfr_daily) / beta

        return treynor_ratio

    def higher_partial_moment(self, mar=0.03, moment=3):

        mar_daily = self._rate_conversion(mar)

        days = self.portfolio_returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.max(self.portfolio_returns - mar_daily, 0), moment)
        )

        return higher_partial_moment

    def lower_partial_moment(self, mar=0.03, moment=3):

        mar_daily = self._rate_conversion(mar)
        days = self.portfolio_returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.max(mar_daily - self.portfolio_returns, 0), moment)
        )

        return lower_partial_moment

    def kappa(self, mar=0.03, moment=3, daily=False, compounding=True):

        lower_partial_moment = self.lower_partial_moment(mar, moment)

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            kappa_ratio = (
                100
                * (self.geometric_mean - mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif not daily and not compounding:
            kappa_ratio = (
                100
                * (self.arithmetic_mean - mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif daily:
            mar_daily = (mar + 1) ** (1 / 252) - 1
            kappa_ratio = (
                100
                * (self.daily_mean - mar_daily)
                / np.power(lower_partial_moment, (1 / moment))
            )

        return kappa_ratio

    def gain_loss(self, mar=0.03, moment=1):

        hpm = self.higher_partial_moment(mar, moment)
        lpm = self.lower_partial_moment(mar, moment)

        gain_loss_ratio = hpm / lpm

        return gain_loss_ratio

    def calmar(self, period=1000, rfr=0.02, daily=False, compounding=True):

        if period >= self.portfolio_state.shape[0]:
            period = self.portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        maximum_drawdown = self.maximum_drawdown_percentage(period)

        if not daily and compounding:
            calmar_ratio = 100 * (self.geometric_mean - rfr) / maximum_drawdown
        if not daily and not compounding:
            calmar_ratio = 100 * (self.arithmetic_mean - rfr) / maximum_drawdown
        elif daily:
            rfr_daily = (rfr + 1) ** (
                1 / 252
            ) - 1  # TODO: find more of these that i might have missed out
            # ? do these assume daily data?
            # TODO: make it general
            calmar_ratio = 100 * (self.daily_mean - rfr_daily) / maximum_drawdown

        return calmar_ratio

    def sterling(self, rfr=0.02, drawdowns=3, daily=False, compounding=True):

        portfolio_drawdowns = self.drawdowns()
        sorted_drawdowns = np.sort(portfolio_drawdowns)
        d_average_drawdown = np.mean(sorted_drawdowns[-drawdowns:])

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            sterling_ratio = (
                100 * (self.geometric_mean - rfr) / np.abs(d_average_drawdown)
            )
        if not daily and not compounding:
            sterling_ratio = (
                100 * (self.arithmetic_mean - rfr) / np.abs(d_average_drawdown)
            )
        elif daily:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sterling_ratio = (
                100 * (self.daily_mean - rfr_daily) / np.abs(d_average_drawdown)
            )

        return sterling_ratio


class Ulcer(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

    def ulcer(self, period=14, start=1):

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
        daily=False,
        compounding=True,
        rfr=0.02,
        period=14,
    ):

        ulcer_index = self.ulcer(period)

        if daily and compounding:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif not daily and compounding:
            martin_ratio = 100 * (self.geometric_mean - rfr) / ulcer_index
        if not daily and not compounding:
            martin_ratio = 100 * (self.arithmetic_mean - rfr) / ulcer_index
        elif daily:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            martin_ratio = 100 * (self.daily_mean - rfr_daily) / ulcer_index

        return martin_ratio

    def ulcer_series(self, period=14):

        ulcer_series = pd.DataFrame(
            columns=["Ulcer Index"], index=self.portfolio_state.index
        )
        for i in range(self.portfolio_state.shape[0] - period):
            ulcer_series.iloc[-i]["Ulcer Index"] = self.ulcer(period, start=i)

        return ulcer_series

    def plot_ulcer(self, period=14, show=True, save=False):

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
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

    def analytical_var(self, value, dof, compounding=True, distribution="normal"):

        if compounding:
            mean_return = self.geometric_mean
        else:
            mean_return = self.arithmetic_mean

        if distribution == "normal":
            var = stats.norm(mean_return, self.volatility).cdf(value)
            expected_loss = (
                stats.norm(mean_return, self.volatility).pdf(
                    stats.norm(mean_return, self.volatility).ppf((1 - var))
                )
                * self.volatility
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
                * self.volatility
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
        compounding=True,
        z=3,
        distribution="Normal",
        show=True,
        save=False,
    ):

        if compounding:
            mean_return = self.geometric_mean
        else:
            mean_return = self.arithmetic_mean

        x = np.linspace(
            mean_return - z * self.volatility, mean_return + z * self.volatility, 100
        )

        if distribution == "Normal":
            pdf = stats.norm(mean_return, self.volatility).pdf(x)
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
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Density of Daily Returns")
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
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Frequency of Daily Returns")
        ax.set_title("Historical Return Distribution and VaR Plot")
        if save:
            plt.savefig("historical_var.png", dpi=300)
        if show:
            plt.show()


class Matrices(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

    def correlation_matrix(self):

        matrix = self.portfolio_returns.corr().round(5)

        return matrix

    def plot_correlation_matrix(self, show=True, save=False):

        matrix = self.correlation_matrix()

        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
        if save:
            plt.savefig("correlation_matrix.png", dpi=300)
        if show:
            plt.show()

    def covariance_matrix(self, daily=True):

        if not daily:  # TODO: dont assume daily
            matrix = self.portfolio_returns.cov().round(5) * self.frequency
        else:
            matrix = self.portfolio_returns.cov().round(5)

        return matrix

    def plot_covariance_matrix(self, daily=True, show=True, save=False):

        matrix = self.covariance_matrix(daily)

        sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
        if save:
            plt.savefig("covariance_matrix.png", dpi=300)
        if show:
            plt.show()


class Utils(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
    ) -> None:

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

    def concatenate_portfolios(self, portfolio_one, portfolio_two):
        """
        Concatenates an array of portfolio returns to an existing array of portfolio returns.
        Accepts array-like objects such as np.ndarray, pd.DataFrame, pd.Series, list etc.

        Args:
            portfolio_one (array-like object): Returns of first portfolio(s).
            portfolio_two (array-like object): Returns of portfolio(s) to be concatenated to the right.

        Returns:
            pd.DataFrame: DataFrame with returns of the given portfolios in respective columns.
        """

        portfolios = pd.concat(
            [pd.DataFrame(portfolio_one), pd.DataFrame(portfolio_two)], axis=1
        )

        return portfolios

    def periodization(self, given_rate, periods=1 / 252):
        """
        Changes rate given in one periodization into another periodization.
        E.g. annual rate of return into daily rate of return etc.

        Args:
            given_rate (float): Rate of interest, return etc. Specified in decimals.
            periods (float, optional): How many given rate periods there is in one output rate period.
                                       Defaults to 1/252. Converts annual rate into daily rate given 252 trading days.
                                       periods=1/365 converts annual rate into daily (calendar) rate.
                                       periods=252 converts daily (trading) rate into annual rate.
                                       periods=12 converts monthly rate into annual.

        Returns:
            float: Rate expressed in a specified period.
        """

        output_rate = (given_rate + 1) ** (periods) - 1

        return output_rate

    def fill_nan(self, portfolio_returns, method="adjacent", data_object="pandas"):

        if data_object == "numpy":
            portfolio_returns = pd.DataFrame(portfolio_returns)

        if method == "adjacent":
            portfolio_returns.interpolate(method="linear", inplace=True)
        elif method == "column":
            portfolio_returns.fillna(portfolio_returns.mean(), inplace=True)
        else:
            raise ValueError("Fill method unsupported.")


class OmegaAnalysis(PortfolioAnalytics):
    def __init__(
        self,
        data,
        weights,
        portfolio_name="Investment Portfolio",
        initial_aum=10000,
        frequency=252,
        mar_lower_bound=0,
        mar_upper_bound=0.2,
    ):
        """
        Initiates the object.

        Args:
            data (pd.DataFrame): Prices data for all assets in portfolio.
            weights (list-like object): Asset weights in portfolio.
            portfolio_name (str, optional): Name of the innvestment portfolio being analysed. Defaults to "Investment Portfolio".
            initial_aum (int, optional): _description_. Defaults to 10000.
            mar_lower_bound (int, optional): MAR lower bound for the Omega Curve. Defaults to 0.
            mar_upper_bound (float, optional): MAR upper bound for the Omega Curve. Defaults to 0.2.
        """

        super.__init__(data, weights, portfolio_name, initial_aum, frequency)

        self.mar_array = np.linspace(
            mar_lower_bound,
            mar_upper_bound,
            round(100 * (mar_upper_bound - mar_lower_bound)),
        )

    def omega_ratio(self, mar=0.03):
        """
        Calculates the Omega Ratio of one or more portfolios.

        Args:
            mar (float, optional): Minimum Acceptable Return. Defaults to 0.03.

        Returns:
            pd.Series: Series with Omega Ratios of all portfolios.
        """  # TODO: change docstrings style

        mar_daily = (mar + 1) ** np.sqrt(1 / 252) - 1

        excess_returns = self.portfolio_returns - mar_daily
        winning = excess_returns[excess_returns > 0].sum()
        losing = -(excess_returns[excess_returns <= 0].sum())

        omega = winning / losing

        return omega

    def omega_curve(self, show=True, save=False):
        """
        Plots and/or saves Omega Curve(s) of of one or more portfolios.

        Args:
            show (bool, optional): Show the plot upon the execution of the code. Defaults to True.
            save (bool, optional): Save the plot on storage. Defaults to False.
        """

        all_values = pd.DataFrame(columns=self.portfolio_returns.columns)

        for portfolio in self.portfolio_returns.columns:
            omega_values = list()
            for mar in self.mar_array:
                value = np.round(
                    self.omega_ratio(self.portfolio_returns[portfolio], mar), 5
                )
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
