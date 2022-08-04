import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self,
                 tickers,
                 weights,
                 start="1970-01-01",
                 end=str(datetime.now())[0:10],
                 data=None,
                 initial_aum=10000,
                 mar=0.03,
                 rfr=0.03):

        self.tickers=tickers
        self.start=start
        self.end=end
        self.data=data   # takes in only the data of kind GetData.save_adj_close_only()
        self.initial_aum = initial_aum

        self.mar_daily = (mar + 1)**np.sqrt(1/252)-1
        self.rfr_daily = (rfr + 1)**np.sqrt(1/252)-1

        if data is None:
            self.prices = pd.DataFrame(columns=self.tickers)
            self.securities_returns = pd.DataFrame(columns=self.tickers)

            for ticker in self.tickers:
                price_current = yf.download(ticker, start=self.start, end=self.end)
                self.prices[ticker] = price_current["Adj Close"]
                self.securities_returns[ticker] = price_current["Adj Close"].pct_change()
        else:
            self.prices = pd.read_csv(self.data, index_col=["Date"])
            self.securities_returns = pd.DataFrame(columns=self.tickers, index=self.prices.index)
            self.securities_returns = self.prices.pct_change()
            self.start = self.prices.index[0]
            self.end = self.prices.index[-1]

        self.securities_returns = self.securities_returns.drop(self.securities_returns.index[0])

        # funds allocated to each security
        self.funds_allocation = np.multiply(self.initial_aum, weights)

        # number of securities bought at t0
        self.initial_number_of_securities_bought = np.divide(self.funds_allocation, self.prices.iloc[0])

        # absolute (dollar) value of each security in portfolio (i.e. state of the portfolio, not rebalanced)
        self.portfolio_state = np.multiply(self.prices, self.initial_number_of_securities_bought)
        self.portfolio_state = pd.DataFrame(self.portfolio_state)
        self.portfolio_state["Whole Portfolio"] = self.portfolio_state.sum(axis=1)

        # portfolio returns (numpy array)
        self.portfolio_returns = np.dot(self.securities_returns.to_numpy(), weights)
        self.mean_returns = np.nanmean(self.portfolio_returns)
        self.volatility = np.nanstd(self.portfolio_returns)


    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def returns_with_dates(self):

        returns_with_dates = pd.DataFrame(self.portfolio_returns, columns=["Returns"], index=self.securities_returns.index)

        return returns_with_dates

    def portfolio_cumulative_returns(self):
        portfolio_returns = self.returns_with_dates()
        portfolio_cumulative_returns = (portfolio_returns["Returns"] + 1).cumprod()
        return portfolio_cumulative_returns

    def plot_portfolio_returns(self,
                               show=True,
                               save=False):

        portfolio_returns = self.returns_with_dates()

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns["Returns"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        if save is True:
            plt.savefig("portfolio_returns.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_returns_distribution(self,
                                            show=True,
                                            save=False):

        portfolio_returns = self.returns_with_dates()

        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Portfolio Returns Distribution")
        if save is True:
            plt.savefig("portfolio_returns_distribution.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_cumulative_returns(self,
                                          show=True,
                                          save=False):

        portfolio_cumulative_returns = self.portfolio_cumulative_returns()

        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_cumulative_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("Portfolio Cumulative Returns")
        if save is True:
            plt.savefig("portfolio_cumulative_returns.png", dpi=300)
        if show is True:
            plt.show()

    def excess_return(self):

        excess_return = self.portfolio_returns - self.mar_daily
        return excess_return

    def benchmark(self,
                  benchmark_tickers,
                  benchmark_weights,
                  data=None):

        if data is None:
            benchmark_securities_prices = pd.DataFrame(columns=benchmark_tickers)
            benchmark_securities_returns = pd.DataFrame(columns=benchmark_tickers)

            for ticker in benchmark_tickers:
                price_current = yf.download(ticker, start=self.start, end=self.end)
                benchmark_securities_prices[ticker] = price_current["Adj Close"]
                benchmark_securities_returns[ticker] = price_current["Adj Close"].pct_change()

            benchmark_returns = np.dot(benchmark_securities_returns.to_numpy(), benchmark_weights)
        else:
            benchmark_securities_prices = pd.read_csv(data, index_col=["Date"])  # takes in only the data of kind GetData.save_adj_close_only()
            benchmark_securities_returns = pd.DataFrame(columns=benchmark_tickers, index=benchmark_securities_prices.index)
            benchmark_securities_returns = benchmark_securities_prices.pct_change()

            benchmark_returns = np.dot(benchmark_securities_returns.to_numpy(), benchmark_weights)

        benchmark_returns = np.delete(benchmark_returns, [0], axis=0)

        return benchmark_returns

    def capm(self,
             benchmark_tickers,
             benchmark_weights,
             data=None):

        excess_portfolio_returns = self.portfolio_returns - self.rfr_daily
        excess_benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        excess_benchmark_returns = excess_benchmark_returns - self.rfr_daily

        model = LinearRegression().fit(excess_benchmark_returns, excess_portfolio_returns)
        alpha = model.intercept_
        beta = model.coef_[0]
        r_squared = model.score(excess_benchmark_returns, excess_portfolio_returns)

        return alpha, beta, r_squared, excess_portfolio_returns, excess_benchmark_returns

    def plot_capm(self,
                  benchmark_tickers,
                  benchmark_weights,
                  data=None,
                  show=True,
                  save=False):

        capm = self.capm(benchmark_tickers, benchmark_weights, data)

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.scatter(capm[4], capm[3], color="b")
        ax1.plot(capm[4], capm[0]+capm[1]*capm[4], color="r")
        empty_patch=mpatches.Patch(color='none', visible=False)
        ax1.legend(handles=[empty_patch, empty_patch],
                   labels=[r"$\alpha$"+" = "+str(np.round(capm[0], 3)),
                           r"$\beta$"+" = "+str(np.round(capm[1], 3))])
        ax1.set_xlabel("Benchmark Excess Returns")
        ax1.set_ylabel("Portfolio Excess Returns")
        ax1.set_title("Portfolio Excess Returns Against Benchmark")
        if save is True:
            plt.savefig("capm.png", dpi=300)
        if show is True:
            plt.show()

    def sharpe(self):
        sharpe_ratio = 100*(self.mean_returns - self.rfr_daily)/self.volatility
        return sharpe_ratio

    def upside_volatility(self):
        positive_portfolio_returns = self.portfolio_returns - self.rfr_daily
        positive_portfolio_returns = positive_portfolio_returns[positive_portfolio_returns>0]
        upside_volatility = np.std(positive_portfolio_returns)
        return upside_volatility

    def downside_volatility(self,
                            returns=None):

        if returns is None:
            negative_portfolio_returns = self.portfolio_returns - self.mar_daily
        else:
            negative_portfolio_returns = returns - self.mar_daily

        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]
        downside_volatility = np.std(negative_portfolio_returns)
        return downside_volatility

    def volatility_skewness(self):
        upside = self.upside_volatility()
        downside = self.downside_volatility()
        skewness = upside/downside
        return skewness

    def omega_excess_return(self,
                            benchmark_tickers,
                            benchmark_weights,
                            data=None):

        portfolio_downside_volatility = self.downside_volatility()
        benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        benchmark_downside_volatility = self.downside_volatility(benchmark_returns)

        omega_excess_return = self.portfolio_returns - 3*portfolio_downside_volatility*benchmark_downside_volatility
        return omega_excess_return

    def upside_potential_ratio(self):
        downside_volatility = self.downside_volatility()
        upside = self.portfolio_returns - self.rfr_daily
        upside = upside[upside>0].sum()
        upside_potential_ratio = upside/downside_volatility
        return upside_potential_ratio

    def downside_capm(self,
                      benchmark_tickers,
                      benchmark_weights,
                      data=None):

        negative_benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        negative_benchmark_returns = negative_benchmark_returns - self.rfr_daily
        negative_benchmark_returns = negative_benchmark_returns[negative_benchmark_returns<0]

        negative_portfolio_returns = self.portfolio_returns - self.rfr_daily
        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]

        model = LinearRegression().fit(negative_benchmark_returns, negative_portfolio_returns)
        downside_alpha = model.intercept_
        downside_beta = model.coef_[0]
        downside_r_squared = model.score(negative_benchmark_returns, negative_portfolio_returns)

        return downside_beta, downside_alpha, downside_r_squared

    def downside_volatility_ratio(self,
                                  benchmark_tickers,
                                  benchmark_weights,
                                  data=None):

        benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)

        portfolio_downside_volatility = self.downside_volatility()
        benchmark_downside_volatility = self.downside_volatility(benchmark_returns)

        downside_volatility_ratio = portfolio_downside_volatility/benchmark_downside_volatility
        return downside_volatility_ratio

    def sortino(self):
        downside_volatility = self.downside_volatility()
        sortino_ratio = 100*(self.mean_returns - self.rfr_daily)/downside_volatility
        return sortino_ratio

    def drawdowns(self):
        wealth_index = 1000*(1+self.returns_with_dates()).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        return drawdowns

    def maximum_drawdown(self,
                         period=1000):

        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown = trough-peak
        return maximum_drawdown

    def maximum_drawdown_percentage(self,
                                    period=1000):

        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown_ratio = (trough-peak)/peak
        return maximum_drawdown_ratio

    def ulcer(self,
              period=14,
              start=1):

        close = np.empty(period)
        percentage_drawdown = np.empty(period)

        if start==1:
            period_high = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        else:
            period_high = np.max(self.portfolio_state.iloc[-period-start+1:-start+1]["Whole Portfolio"])

        for i in range(period):
            close[i] = self.portfolio_state.iloc[-i-start+1]["Whole Portfolio"]
            percentage_drawdown[i] = 100*((close[i] - period_high))/period_high

        ulcer_index = np.sqrt(np.mean(np.square(percentage_drawdown)))
        return ulcer_index

    def plot_ulcer(self,
                   period=14,
                   show=True,
                   save=False):

        ulcer_values = pd.DataFrame(columns=["Ulcer Index"], index=self.portfolio_state.index)
        for i in range(self.portfolio_state.shape[0]-period):
            ulcer_values.iloc[-i]["Ulcer Index"] = self.ulcer(period, start=i)

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ulcer_values["Ulcer Index"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Ulcer Index")
        ax1.set_title("Portfolio Ulcer Index")
        if save is True:
            plt.savefig("ulcer.png", dpi=300)
        if show is True:
            plt.show()

    def martin(self,
               period=14):

        ulcer_index = self.ulcer(period)
        sharpe_ratio = 100*(self.mean_returns - self.rfr_daily)/ulcer_index
        return sharpe_ratio

    def omega_ratio(self):

        returns = self.returns_with_dates()

        excess_returns = returns-self.mar_daily
        winning = excess_returns[excess_returns>0].sum()
        losing = -(excess_returns[excess_returns<=0].sum())

        omega=winning/losing
        return omega

    def jensen_alpha(self,
                     benchmark_tickers,
                     benchmark_weights,
                     data=None):

        capm = self.capm(benchmark_tickers, benchmark_weights, data)

        jensen_alpha = self.mean_returns - self.rfr_daily - capm[1](np.nanmean(capm[4]) - self.rfr_daily)

        return jensen_alpha

    def treynor(self,
                benchmark_tickers,
                benchmark_weights,
                data=None):

        capm = self.capm(benchmark_tickers, benchmark_weights, data)

        treynor_ratio = 100*(self.mean_returns-self.rfr_daily)/capm[1]

        return treynor_ratio

    def higher_partial_moment(self,
                              moment=3):

        days = self.portfolio_returns.shape[0]

        higher_partial_moment = (1/days)*np.sum(np.power(np.max(self.portfolio_returns-self.mar_daily, 0), moment))

        return higher_partial_moment

    def lower_partial_moment(self,
                             moment=3):

        days = self.portfolio_returns.shape[0]

        lower_partial_moment = (1/days)*np.sum(np.power(np.max(self.mar_daily-self.portfolio_returns, 0), moment))

        return lower_partial_moment

    def kappa(self,
              moment=3):

        lower_partial_moment = self.lower_partial_moment(moment)

        kappa_ratio = (self.mean_returns - self.rfr_daily)/np.power(lower_partial_moment, (1/moment))

        return kappa_ratio

    def calmar(self,
               period=1000):
        maximum_drawdown = self.maximum_drawdown_percentage(period)
        calmar_ratio = self.mean_returns-self.rfr_daily/maximum_drawdown

        return calmar_ratio

    def sterling(self,
                 drawdowns=3):

        portfolio_drawdowns = self.drawdowns()
        sorted_drawdowns = np.argsort(portfolio_drawdowns)
        d_average_drawdown = np.mean(sorted_drawdowns[-drawdowns:])

        sterling_ratio = (self.portfolio_returns - self.rfr_daily)/np.abs(d_average_drawdown)

        return sterling_ratio

    def gain_loss(self,
                  mar=0.03,
                  moment=1):

        hpm=self.higher_partial_moment(moment)
        lpm=self.lower_partial_moment(moment)

        gain_loss_ratio = hpm/lpm

        return gain_loss_ratio

    def tracking_error(self,
                       benchmark_tickers,
                       benchmark_weights,
                       data=None):

        benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)

        tracking_error = np.std(self.portfolio_returns, benchmark_returns)

        return tracking_error

    def correlation_matrix(self,
                           show=True,
                           save=False):

        matrix=self.prices.corr().round(2)

        sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
        if save is True:
            plt.savefig("correlation_matrix.png", dpi=300)
        if show is True:
            plt.show()

        return matrix

    def covariance_matrix(self,
                          show=True,
                          save=False):

        matrix=self.prices.cov().round(2)

        sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
        if save is True:
            plt.savefig("covariance_matrix.png", dpi=300)
        if show is True:
            plt.show()

        return matrix

    def analytical_var(self,
                       value,
                       dof,
                       distribution="normal"):

        if distribution=="normal":
            var=stats.norm(self.mean_returns, self.volatility).cdf(value)
            expected_loss=(stats.norm(self.mean_returns, self.volatility). \
                          pdf(stats.norm(self.mean_returns, self.volatility). \
                          ppf((1 - var))) * self.volatility)/(1 - var) - self.mean_returns

        elif distribution=="t":
            var=stats.t(dof).cdf(value)
            percent_point_function = stats.t(dof).ppf((1 - var))
            expected_loss = -1/(1 - var)*(1-dof)**(-1)*(dof-2 + percent_point_function**2) \
                            *stats.t(dof).pdf(percent_point_function)*self.volatility-self.mean_returns

        else:
            print("Distribution Unavailable.")

        return var, expected_loss

    def historical_var(self,
                       value):

        returns_below_value = self.portfolio_returns[self.portfolio_returns<value]
        var=returns_below_value.shape[0]/self.portfolio_returns.shape[0]

        return var

    def plot_analytical_var(self,
                            value,
                            dof,
                            z=3,
                            distribution="Normal",
                            show=True,
                            save=False):

        x = np.linspace(self.mean_returns-z*self.volatility, self.mean_returns+z*self.volatility, self.portfolio_returns.shape[0])

        if distribution=="Normal":
            pdf=stats.norm(self.mean_returns, self.volatility).pdf(x)
        elif distribution=="t":
            pdf=stats.t(dof).pdf(x)
        else:
            print("Distribution Unavailable")

        cutoff = (np.abs(x - value)).argmin()

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.plot(x, pdf, linewidth=2, color="b", label="Analytical (Theoretical) Distribution of Portfolio Returns")
        ax1.fill_between(x[0:cutoff], pdf[0:cutoff], facecolor="r", label="Analytical VaR")
        ax1.legend()
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Density of Daily Returns")
        ax1.set_title("Analytical (Theoretical," + distribution + ") Return Distribution and VaR Plot")
        if save is True:
            plt.savefig("analytical_var.png", dpi=300)
        if show is True:
            plt.show()

    def plot_historical_var(self,
                            value,
                            number_of_bins,
                            show=True,
                            save=False):

        sorted_portfolio_returns = np.sort(self.portfolio_returns)
        bins = np.linspace(sorted_portfolio_returns[0], sorted_portfolio_returns[-1]+1, number_of_bins)

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.hist(sorted_portfolio_returns, bins, label="Historical Distribution of Portfolio Returns")
        ax1.axvline(x=value, ymin=0, color="r", label="Historical VaR Cutoff")
        ax1.legend()
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Frequency of Daily Returns")
        ax1.set_title("Historical Return Distribution and VaR Plot")
        if save is True:
            plt.savefig("historical_var.png", dpi=300)
        if show is True:
            plt.show()

# TODO: same analytics for each security in the portfolio separately
# TODO: as other class: benchmarking, PMPT, MPT, etc
# TODO: separate one for checking which weigths
# TODO: split them into classes such as they are grouped by what they are dependant on

#TODO: backtesting class

# TODO: do portfolio metrics in one method (__init__) OR do a complete report() function

# __name__==__main__
# make var names smaller
# pytest


# ? annualizing ratios?
# period=len(data) | x*np.sqrt(period)

# cloud providers, chip names (SOXX), cyber security
