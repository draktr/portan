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

# TODO: black
# TODO: other cov matrices (ledoit-wolf etc)
# TODO: other methods of returns
# TODO: checker methods
# TODO: handling of missing data and series of different lengths
# TODO: analytics get saved in the object as they get executed


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
        self.portfolio_state = np.multiply(self.prices, self.allocation_assets)
        self.portfolio_state["Whole Portfolio"] = self.portfolio_state.sum(axis=1)

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

    def excess_returns(self, portfolio_returns=None, mar=0.03):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        mar_daily = (mar + 1) ** (1 / 252) - 1
        excess_returns = portfolio_returns - mar_daily

        if portfolio_returns is None:
            self.analytics.update({str(inspect.stack()[0][3]): excess_returns})

        return excess_returns

    def net_return(
        self,
        allocation_assets=None,
        assets_info=None,
        initial_aum=None,
        percentage=False,
    ):

        if initial_aum is None:
            initial_aum = self.initial_aum

        final_aum = self.final_aum(allocation_assets, assets_info)

        if percentage is False:
            net_return = final_aum - initial_aum
        elif percentage is True:
            net_return = (final_aum - initial_aum) / initial_aum
        else:
            raise ValueError("Argument 'percentage' has to be boolean.")

        if allocation_assets is None:
            self.analytics.update({str(inspect.stack()[0][3]): net_return})

        return net_return

    def min_aum(self, portfolio_state=None):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        min_aum = portfolio_state["Whole Portfolio"].min()

        if portfolio_state is None:
            self.analytics.update({str(inspect.stack()[0][3]): min_aum})

        return min_aum

    def max_aum(self, portfolio_state=None):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        max_aum = portfolio_state["Whole Portfolio"].max()

        if portfolio_state is None:
            self.analytics.update({str(inspect.stack()[0][3]): max_aum})

        return max_aum

    def mean_aum(self, portfolio_state=None):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        mean_aum = portfolio_state["Whole Portfolio"].mean()

        if portfolio_state is None:
            self.analytics.update({str(inspect.stack()[0][3]): mean_aum})

        return mean_aum

    def final_aum(self, allocation_assets=None, assets_info=None):

        if allocation_assets is None:
            allocation_assets = self.allocation_assets
        if assets_info is None:
            assets_info = self.assets_info

        final_aum = allocation_assets * assets_info["regularMarketPrice"]

        if allocation_assets is None:
            setattr(
                self, self.analytics, self.analytics.update({"final_aum": final_aum})
            )

        return final_aum

    def distribution_test(
        self, portfolio_returns=None, test="dagostino-pearson", distribution="norm"
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        if test == "kolomogorov-smirnov":
            result = stats.kstest(portfolio_returns, distribution)
        elif test == "lilliefors":
            result = lilliefors(portfolio_returns)
        elif test == "shapiro-wilk":
            result = stats.shapiro(portfolio_returns)
        elif test == "jarque-barre":
            result = stats.jarque_bera(portfolio_returns)
        elif test == "dagostino-pearson":
            result = stats.normaltest(portfolio_returns)
        elif test == "anderson-darling":
            result = stats.anderson(portfolio_returns, distribution)
        else:
            raise ValueError("Statistical test is unavailable.")

        if portfolio_returns is None:
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

    def plot_aum(self, portfolio_state=None, show=True, save=False):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        portfolio_state["Whole Portfolio"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("AUM ($)")
        ax.set_title("Assets Under Management")
        if save is True:
            plt.savefig("aum.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_returns(self, portfolio_returns=None, show=True, save=False):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        portfolio_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Returns")
        ax.set_title("Portfolio Daily Returns")
        if save is True:
            plt.savefig("portfolio_returns.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_returns_distribution(
        self, portfolio_returns=None, show=True, save=False
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        portfolio_returns.plot.hist(bins=90)
        ax.set_xlabel("Daily Returns")
        ax.set_ylabel("Frequency")
        ax.set_title("Portfolio Returns Distribution")
        if save is True:
            plt.savefig("portfolio_returns_distribution.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_cumulative_returns(
        self, portfolio_cumulative_returns=None, show=True, save=False
    ):

        if portfolio_cumulative_returns is None:
            portfolio_cumulative_returns = self.portfolio_cumulative_returns

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        portfolio_cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Portfolio Cumulative Returns")
        if save is True:
            plt.savefig("portfolio_cumulative_returns.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_piechart(
        self,
        weights,
        initial_aum=None,
        tickers=None,
        assets_names=None,
        portfolio_name=None,
        show=True,
        save=False,
    ):
        if initial_aum is None:
            initial_aum = self.initial_aum
        if tickers is None:
            tickers = self.tickers
        if assets_names is None:
            assets_names = self.assets_names
        if portfolio_name is None:
            portfolio_name = self.portfolio_name

        allocation_funds = np.multiply(initial_aum, weights)
        wp = {"linewidth": 1, "edgecolor": "black"}
        explode = tuple(repeat(0.05, len(tickers)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        pie = ax.pie(
            allocation_funds,
            autopct=lambda pct: self._ap(pct, allocation_funds),
            explode=explode,
            labels=tickers,
            shadow=True,
            startangle=90,
            wedgeprops=wp,
        )
        ax.legend(
            pie[0],
            assets_names,
            title="Portfolio Assets",
            loc="upper right",
            bbox_to_anchor=(0.7, 0, 0.5, 1),
        )
        plt.setp(pie[2], size=9, weight="bold")
        ax.set_title(str(portfolio_name + " Asset Distribution"))
        if save is True:
            plt.savefig(str(portfolio_name + "_pie_chart.png"), dpi=300)
        if show is True:
            plt.show()

    def plot_assets_cumulative_returns(
        self, assets_returns=None, assets_names=None, show=True, save=False
    ):

        if assets_returns is None:
            assets_returns = self.assets_returns
        if assets_names is None:
            assets_names = self.assets_names

        assets_cumulative_returns = (assets_returns + 1).cumprod()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        assets_cumulative_returns.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.set_title("Assets Cumulative Returns")
        ax.legend(labels=assets_names)
        if save is True:
            plt.savefig("assets_cumulative_returns.png", dpi=300)
        if show is True:
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

    def capm(self, portfolio_returns=None, rfr=0.02):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        rfr_daily = (rfr + 1) ** (1 / 252) - 1

        excess_portfolio_returns = portfolio_returns - rfr_daily
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

    def plot_capm(self, portfolio_returns=None, rfr=0.02, show=True, save=False):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        capm = self.capm(portfolio_returns, rfr)

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
        if save is True:
            plt.savefig("capm.png", dpi=300)
        if show is True:
            plt.show()

    def sharpe(
        self,
        portfolio_returns=None,
        daily=False,
        compounding=True,
        rfr=0.02,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
            volatility = self.volatility
            daily_volatility = self.daily_volatility
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1
            daily_volatility = portfolio_returns.std()
            volatility = daily_volatility * np.sqrt(frequency)

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            sharpe_ratio = 100 * (geometric_mean - rfr) / volatility
        if daily is False and compounding is False:
            sharpe_ratio = 100 * (arithmetic_mean - rfr) / volatility
        elif daily is True:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sharpe_ratio = 100 * (daily_mean - rfr_daily) / daily_volatility

        return sharpe_ratio

    def tracking_error(self, portfolio_returns=None):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        tracking_error = np.std(portfolio_returns - self.benchmark_returns, ddof=1)

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

    def upside_volatility(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        mar_daily = (mar + 1) ** (1 / 252) - 1

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        positive_portfolio_returns = portfolio_returns - mar_daily
        positive_portfolio_returns = positive_portfolio_returns[
            positive_portfolio_returns > 0
        ]
        if daily is False:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1) * frequency
        else:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1)

        return upside_volatility

    def downside_volatility(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        mar_daily = (mar + 1) ** (1 / 252) - 1

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        negative_portfolio_returns = portfolio_returns - mar_daily
        negative_portfolio_returns = negative_portfolio_returns[
            negative_portfolio_returns < 0
        ]
        if daily is False:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1) * frequency
        else:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1)

        return downside_volatility

    def volatility_skew(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        upside = self.upside_volatility(portfolio_returns, mar, daily, frequency)
        downside = self.downside_volatility(portfolio_returns, mar, daily, frequency)
        skew = upside / downside

        return skew

    def omega_excess_return(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        portfolio_downside_volatility = self.downside_volatility(
            portfolio_returns, mar, daily, frequency
        )
        benchmark_downside_volatility = self.downside_volatility(
            self.benchmark_returns, mar, daily, self.frequency
        )

        omega_excess_return = (
            portfolio_returns
            - 3 * portfolio_downside_volatility * benchmark_downside_volatility
        )

        return omega_excess_return

    def upside_potential_ratio(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        mar_daily = (mar + 1) ** (1 / 252) - 1

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        downside_volatility = self.downside_volatility(
            portfolio_returns, mar, daily, frequency
        )
        upside = portfolio_returns - mar_daily
        upside = upside[upside > 0].sum()
        upside_potential_ratio = upside / downside_volatility

        return upside_potential_ratio

    def downside_capm(self, portfolio_returns=None, mar=0.03):

        mar_daily = (mar + 1) ** (1 / 252) - 1

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        negative_benchmark_returns = self.benchmark_returns - mar_daily
        negative_benchmark_returns = negative_benchmark_returns[
            negative_benchmark_returns < 0
        ]

        negative_portfolio_returns = portfolio_returns - mar_daily
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

    def downside_volatility_ratio(
        self, portfolio_returns=None, mar=0.03, daily=False, frequency=252
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            frequency = self.frequency

        portfolio_downside_volatility = self.downside_volatility(
            portfolio_returns, mar, daily, frequency
        )
        benchmark_downside_volatility = self.downside_volatility(
            self.benchmark_returns, mar, daily, self.frequency
        )

        downside_volatility_ratio = (
            portfolio_downside_volatility / benchmark_downside_volatility
        )

        return downside_volatility_ratio

    def sortino(
        self,
        portfolio_returns=None,
        mar=0.03,
        rfr=0.02,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
            frequency = self.frequency
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        downside_volatility = self.downside_volatility(
            portfolio_returns, mar, daily, frequency
        )

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            sortino_ratio = 100 * (geometric_mean - rfr) / downside_volatility
        elif daily is False and compounding is False:
            sortino_ratio = 100 * (arithmetic_mean - rfr) / downside_volatility
        elif daily is True:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sortino_ratio = 100 * (daily_mean - rfr_daily) / downside_volatility

        return sortino_ratio

    def drawdowns(self, portfolio_returns=None):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        wealth_index = 1000 * (1 + portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return drawdowns

    def maximum_drawdown(self, portfolio_state=None, period=1000):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        if period >= portfolio_state.shape[0]:
            period = portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        peak = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = portfolio_state.index.get_loc(peak_index)
        trough = np.min(portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown = trough - peak

        return maximum_drawdown

    def maximum_drawdown_percentage(self, portfolio_state=None, period=1000):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        if period > portfolio_state.shape[0]:
            period = portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        peak = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = portfolio_state.index.get_loc(peak_index)
        trough = np.min(portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown_ratio = (trough - peak) / peak

        return maximum_drawdown_ratio

    def jensen_alpha(
        self,
        portfolio_returns=None,
        rfr=0.02,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        rfr_daily = (rfr + 1) ** (1 / 252) - 1

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        excess_portfolio_returns = portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        beta = model.coef_[0]

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            jensen_alpha = (
                geometric_mean - rfr - beta * (self.benchmark_geometric_mean - rfr)
            )
        if daily is False and compounding is False:
            jensen_alpha = (
                arithmetic_mean - rfr - beta * (self.benchmark_arithmetic_mean - rfr)
            )
        elif daily is True:
            jensen_alpha = (
                daily_mean - rfr_daily - beta * (self.benchmark_daily_mean - rfr_daily)
            )

        return jensen_alpha

    def treynor(
        self,
        portfolio_returns=None,
        rfr=0.02,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        rfr_daily = (rfr + 1) ** (1 / 252) - 1
        excess_portfolio_returns = portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(
            excess_benchmark_returns, excess_portfolio_returns
        )
        beta = model.coef_[0]

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            treynor_ratio = 100 * (geometric_mean - rfr) / beta
        if daily is False and compounding is False:
            treynor_ratio = 100 * (arithmetic_mean - rfr) / beta
        elif daily is True:
            treynor_ratio = 100 * (daily_mean - rfr_daily) / beta

        return treynor_ratio

    def higher_partial_moment(self, portfolio_returns=None, mar=0.03, moment=3):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        mar_daily = (mar + 1) ** (1 / 252) - 1
        days = portfolio_returns.shape[0]

        higher_partial_moment = (1 / days) * np.sum(
            np.power(np.max(portfolio_returns - mar_daily, 0), moment)
        )

        return higher_partial_moment

    def lower_partial_moment(self, portfolio_returns=None, mar=0.03, moment=3):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        mar_daily = (mar + 1) ** (1 / 252) - 1
        days = portfolio_returns.shape[0]

        lower_partial_moment = (1 / days) * np.sum(
            np.power(np.max(mar_daily - portfolio_returns, 0), moment)
        )

        return lower_partial_moment

    def kappa(
        self,
        portfolio_returns=None,
        mar=0.03,
        moment=3,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        lower_partial_moment = self.lower_partial_moment(portfolio_returns, mar, moment)

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            kappa_ratio = (
                100
                * (geometric_mean - mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif daily is False and compounding is False:
            kappa_ratio = (
                100
                * (arithmetic_mean - mar)
                / np.power(lower_partial_moment, (1 / moment))
            )
        elif daily is True:
            mar_daily = (mar + 1) ** (1 / 252) - 1
            kappa_ratio = (
                100
                * (daily_mean - mar_daily)
                / np.power(lower_partial_moment, (1 / moment))
            )

        return kappa_ratio

    def gain_loss(self, portfolio_returns=None, mar=0.03, moment=1):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        hpm = self.higher_partial_moment(portfolio_returns, mar, moment)
        lpm = self.lower_partial_moment(portfolio_returns, mar, moment)

        gain_loss_ratio = hpm / lpm

        return gain_loss_ratio

    def calmar(
        self,
        portfolio_returns=None,
        portfolio_state=None,
        period=1000,
        rfr=0.02,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        if portfolio_returns is None and portfolio_state is not None:
            raise ValueError(
                "Argument portfolio_returns not provided. \
                              Portfolio returns need to be provided for the calculation of mean return."
            )
        elif portfolio_state is None:
            portfolio_state = self.portfolio_state
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        if period >= portfolio_state.shape[0]:
            period = portfolio_state.shape[0]
            warnings.warn("Dataset too small. Period taken as {}.".format(period))

        maximum_drawdown = self.maximum_drawdown_percentage(portfolio_state, period)

        if daily is False and compounding is True:
            calmar_ratio = 100 * (geometric_mean - rfr) / maximum_drawdown
        if daily is False and compounding is False:
            calmar_ratio = 100 * (arithmetic_mean - rfr) / maximum_drawdown
        elif daily is True:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            calmar_ratio = 100 * (daily_mean - rfr_daily) / maximum_drawdown

        return calmar_ratio

    def sterling(
        self,
        portfolio_returns=None,
        rfr=0.02,
        drawdowns=3,
        daily=False,
        compounding=True,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        portfolio_drawdowns = self.drawdowns(portfolio_returns)
        sorted_drawdowns = np.sort(portfolio_drawdowns)
        d_average_drawdown = np.mean(sorted_drawdowns[-drawdowns:])

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            sterling_ratio = 100 * (geometric_mean - rfr) / np.abs(d_average_drawdown)
        if daily is False and compounding is False:
            sterling_ratio = 100 * (arithmetic_mean - rfr) / np.abs(d_average_drawdown)
        elif daily is True:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            sterling_ratio = 100 * (daily_mean - rfr_daily) / np.abs(d_average_drawdown)

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

    def ulcer(self, portfolio_state=None, period=14, start=1):

        close = np.empty(period)
        percentage_drawdown = np.empty(period)

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        if start == 1:
            period_high = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        else:
            period_high = np.max(
                portfolio_state.iloc[-period - start + 1 : -start + 1][
                    "Whole Portfolio"
                ]
            )

        for i in range(period):
            close[i] = portfolio_state.iloc[-i - start + 1]["Whole Portfolio"]
            percentage_drawdown[i] = 100 * ((close[i] - period_high)) / period_high

        ulcer_index = np.sqrt(np.mean(np.square(percentage_drawdown)))

        return ulcer_index

    def martin(
        self,
        portfolio_returns=None,
        daily=False,
        compounding=True,
        rfr=0.02,
        period=14,
        frequency=252,
    ):

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns
            geometric_mean = self.geometric_mean
            arithmetic_mean = self.arithmetic_mean
            daily_mean = self.daily_mean
        else:
            daily_mean = portfolio_returns.mean()
            arithmetic_mean = daily_mean * frequency
            geometric_mean = (1 + portfolio_returns).prod() ** (
                frequency / portfolio_returns.shape[0]
            ) - 1

        ulcer_index = self.ulcer(period)

        if daily is True and compounding is True:
            raise ValueError("Mean returns cannot be compounded if daily.")
        elif daily is False and compounding is True:
            martin_ratio = 100 * (geometric_mean - rfr) / ulcer_index
        if daily is False and compounding is False:
            martin_ratio = 100 * (arithmetic_mean - rfr) / ulcer_index
        elif daily is True:
            rfr_daily = (rfr + 1) ** (1 / 252) - 1
            martin_ratio = 100 * (daily_mean - rfr_daily) / ulcer_index

        return martin_ratio

    def ulcer_series(self, portfolio_state=None, period=14):

        if portfolio_state is None:
            portfolio_state = self.portfolio_state

        ulcer_series = pd.DataFrame(
            columns=["Ulcer Index"], index=portfolio_state.index
        )
        for i in range(portfolio_state.shape[0] - period):
            ulcer_series.iloc[-i]["Ulcer Index"] = self.ulcer(period, start=i)

        return ulcer_series

    def plot_ulcer(self, portfolio_state=None, period=14, show=True, save=False):

        ulcer_series = self.ulcer_series(portfolio_state, period)

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ulcer_series["Ulcer Index"].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ulcer Index")
        ax.set_title("Portfolio Ulcer Index")
        if save is True:
            plt.savefig("ulcer.png", dpi=300)
        if show is True:
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

    def analytical_var(
        self, mean_return, volatility, value, dof, distribution="normal"
    ):

        if distribution == "normal":
            var = stats.norm(mean_return, volatility).cdf(value)
            expected_loss = (
                stats.norm(mean_return, volatility).pdf(
                    stats.norm(mean_return, volatility).ppf((1 - var))
                )
                * volatility
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
                * volatility
                - mean_return
            )
        else:
            raise ValueError("Probability distribution unavailable.")

        return var, expected_loss

    def historical_var(self, portfolio_returns, value):

        returns_below_value = portfolio_returns[portfolio_returns < value]
        var = returns_below_value.shape[0] / portfolio_returns.shape[0]

        return var

    def plot_analytical_var(
        self,
        mean_return,
        volatility,
        value,
        dof,
        z=3,
        distribution="Normal",
        show=True,
        save=False,
    ):

        x = np.linspace(mean_return - z * volatility, mean_return + z * volatility, 100)

        if distribution == "Normal":
            pdf = stats.norm(mean_return, volatility).pdf(x)
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
        if save is True:
            plt.savefig("analytical_var.png", dpi=300)
        if show is True:
            plt.show()

    def plot_historical_var(
        self, portfolio_returns, value, number_of_bins=100, show=True, save=False
    ):

        sorted_portfolio_returns = np.sort(portfolio_returns)
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
        if save is True:
            plt.savefig("historical_var.png", dpi=300)
        if show is True:
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

    def correlation_matrix(self, returns=None):

        if returns is None:
            returns = self.portfolio_returns

        matrix = returns.corr().round(5)

        return matrix

    def plot_correlation_matrix(self, returns=None, show=True, save=False):

        matrix = self.correlation_matrix(returns)

        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
        if save is True:
            plt.savefig("correlation_matrix.png", dpi=300)
        if show is True:
            plt.show()

    def covariance_matrix(self, returns=None, daily=True, frequency=252):

        if returns is None:
            returns = self.portfolio_returns
            frequency = self.frequency

        if daily is False:
            matrix = returns.cov().round(5) * frequency
        else:
            matrix = returns.cov().round(5)

        return matrix

    def plot_covariance_matrix(
        self, returns=None, daily=True, frequency=252, show=True, save=False
    ):

        matrix = self.covariance_matrix(returns, daily, frequency)

        sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
        if save is True:
            plt.savefig("covariance_matrix.png", dpi=300)
        if show is True:
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

    def omega_ratio(self, portfolio_returns=None, mar=0.03):
        """
        Calculates the Omega Ratio of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame): Dataframe with the daily returns of single or more portfolios. Defaults to None
            mar (float, optional): Minimum Acceptable Return. Defaults to 0.03.

        Returns:
            pd.Series: Series with Omega Ratios of all portfolios.
        """

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        mar_daily = (mar + 1) ** np.sqrt(1 / 252) - 1

        excess_returns = portfolio_returns - mar_daily
        winning = excess_returns[excess_returns > 0].sum()
        losing = -(excess_returns[excess_returns <= 0].sum())

        omega = winning / losing

        return omega

    def omega_curve(self, portfolio_returns=None, show=True, save=False):
        """
        Plots and/or saves Omega Curve(s) of of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame): Dataframe with the daily returns of single or more portfolios. Defaults to None
            show (bool, optional): Show the plot upon the execution of the code. Defaults to True.
            save (bool, optional): Save the plot on storage. Defaults to False.
        """

        if portfolio_returns is None:
            portfolio_returns = self.portfolio_returns

        all_values = pd.DataFrame(columns=portfolio_returns.columns)

        for portfolio in portfolio_returns.columns:
            omega_values = list()
            for mar in self.mar_array:
                value = np.round(self.omega_ratio(portfolio_returns[portfolio], mar), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(
            title="Omega Curve",
            xlabel="Minimum Acceptable Return (%)",
            ylabel="Omega Ratio",
            ylim=(0, 1.5),
        )
        if save is True:
            plt.savefig("omega_curves.png", dpi=300)
        if show is True:
            plt.show()
