import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy import optimize
from datetime import datetime
from itertools import repeat


class PortfolioAnalytics():
    def __init__(self,
                 tickers=None,
                 start="1970-01-02",
                 end=str(datetime.now())[0:10],
                 portfolio_name="Investment Portfolio",
                 data=None,     # takes in only the data of kind GetData.save_adj_close_only()
                 initial_aum=10000):

        self.tickers=tickers
        self.portfolio_name=portfolio_name
        self.initial_aum = initial_aum
        """
        self.names = pd.Series(index=self.tickers, dtype="str")
        for ticker in self.tickers:     # TODO: fix this
            security = yf.Ticker(ticker)
            self.names[ticker] = security.info["longName"]
        """

        if data is None:
            self.prices=pdr.DataReader(self.tickers, start=start, end=end, data_source="yahoo")["Adj Close"]
            self.assets_returns=self.prices.pct_change()
        else:
            self.prices = pd.read_csv(data, index_col=["Date"])
            self.assets_returns = self.prices.pct_change()
            self.tickers=self.prices.columns

        self.assets_returns = self.assets_returns.drop(self.assets_returns.index[0])

    def portfolio_state(self,
                        weights):

        # funds allocated to each security
        allocation_funds = np.multiply(self.initial_aum, weights)

        # number of assets bought at t0
        allocation_assets = np.divide(allocation_funds, self.prices.iloc[0])

        # absolute (dollar) value of each security in portfolio (i.e. state of the portfolio, not rebalanced)
        portfolio_state = np.multiply(self.prices, allocation_assets)
        portfolio_state["Whole Portfolio"] = portfolio_state.sum(axis=1)

        return allocation_funds, allocation_assets, portfolio_state

    def portfolio_returns(self,
                          weights,
                          return_dataframe=True):

        portfolio_returns = np.dot(self.assets_returns.to_numpy(), weights)

        if return_dataframe is True:
            portfolio_returns = pd.DataFrame(portfolio_returns,
                                             columns=[self.portfolio_name],
                                             index=self.assets_returns.index)

        return portfolio_returns

    def mean_return(self,
                    portfolio_returns,
                    daily=False,
                    compounding=True,
                    frequency=252):

        if daily is False and compounding is True:
            mean_return = (1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
        elif daily is False and compounding is False:
            mean_return = portfolio_returns.mean()*frequency
        elif daily is True:
            mean_return = portfolio_returns.mean()
        else:
            print("Error: mean returns cannot be compounded if daily")

        return mean_return

    def volatility(self,
                   portfolio_returns,
                   daily=False,
                   frequency=252):

        if daily is False:
            volatility = portfolio_returns.std()*np.sqrt(frequency)
        else:
            volatility = portfolio_returns.std()

        return volatility

    def excess_returns(self,
                       portfolio_returns,
                       mar=0.03):

        mar_daily = (mar + 1)**(1/252)-1
        excess_returns = portfolio_returns - mar_daily

        return excess_returns

    def portfolio_cumulative_returns(self,
                                     portfolio_returns):

        portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()

        return portfolio_cumulative_returns

    def plot_portfolio_returns(self,
                               portfolio_returns,
                               show=True,
                               save=False):

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        if save is True:
            plt.savefig("portfolio_returns.png", dpi=300)
        if show is True:
            plt.show()

    def plot_portfolio_returns_distribution(self,
                                            portfolio_returns,
                                            show=True,
                                            save=False):

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
                                          portfolio_returns,
                                          show=True,
                                          save=False):

        portfolio_cumulative_returns = self.portfolio_cumulative_returns(portfolio_returns)

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

    def plot_portfolio_piechart(self,
                                weights,
                                show=True,
                                save=False):

        allocation_funds = np.multiply(self.initial_aum, weights)
        wp={'linewidth':1, 'edgecolor':"black" }
        explode=tuple(repeat(0.05, len(self.tickers)))

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        pie=ax1.pie(allocation_funds,
                      autopct=lambda pct: self._ap(pct, allocation_funds),
                      explode=explode,
                      labels=self.tickers,
                      shadow=True,
                      startangle=90,
                      wedgeprops=wp)
        ax1.legend(pie[0], self.names,
                   title="Portfolio Assets",
                   loc="upper right",
                   bbox_to_anchor=(0.7, 0, 0.5, 1))
        plt.setp(pie[2], size=9, weight="bold")
        ax1.set_title(str(self.portfolio_name+" Asset Distribution"))
        if save is True:
            plt.savefig(str(self.portfolio_name+"_pie_chart.png"), dpi=300)
        if show is True:
            plt.show()

    def _ap(self, pct, all_values):
        absolute = int(pct / 100.*np.sum(all_values))
        return "{:.1f}%\n(${:d})".format(pct, absolute)

# TODO: rebalancing, transaction costs
# TODO: __name__==__main__
# TODO: pytest
# TODO: plot cumulative returns of all securities in portfolio
# TODO: plot cumulative returns of different protfolios
# TODO: other cov matrices (ledoit-wolf etc)
# TODO: other methods of returns
# TODO: checker methods

"""
if not isinstance(risk_free_rate, (int, float)):
    raise ValueError("risk_free_rate should be numeric")
self._risk_free_rate = risk_free_rate

"""
# separate method for plotting matrices
# ? annualizing ratios?


class MPT():
    def __init__(self,
                  benchmark_weights,
                  benchmark_tickers=None,
                  benchmark_data=None,
                  start="1970-01-01",
                  end=str(datetime.now())[0:10]) -> None:

        if benchmark_data is None:
            benchmark_assets_prices=pdr.DataReader(benchmark_tickers, start=start, end=end, data_source="yahoo")["Adj Close"]
            benchmark_assets_returns=benchmark_assets_prices.pct_change()
        else:
            benchmark_assets_prices = pd.read_csv(benchmark_data, index_col=["Date"])    # takes in only the data of kind GetData.save_adj_close_only()
            benchmark_assets_returns = benchmark_assets_prices.pct_change()

        self.benchmark_returns = np.dot(benchmark_assets_returns.to_numpy(), benchmark_weights)
        self.benchmark_returns = np.delete(self.benchmark_returns, [0], axis=0)

    def capm(self,
             portfolio_returns,
             rfr=0.02):

        rfr_daily = (rfr + 1)**(1/252)-1

        excess_portfolio_returns = portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(excess_benchmark_returns, excess_portfolio_returns)
        alpha = model.intercept_
        beta = model.coef_[0]
        r_squared = model.score(excess_benchmark_returns, excess_portfolio_returns)

        return alpha, beta, r_squared, excess_portfolio_returns, excess_benchmark_returns

    def plot_capm(self,
                  portfolio_returns,
                  rfr=0.02,
                  show=True,
                  save=False):

        capm = self.capm(portfolio_returns, rfr)

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
        ax1.set_title("Portfolio Excess Returns Against Benchmark (CAPM)")
        if save is True:
            plt.savefig("capm.png", dpi=300)
        if show is True:
            plt.show()

    def sharpe(self,
               mean_return,
               volatility,
               rfr=0.02):

        sharpe_ratio = 100*(mean_return - rfr)/volatility

        return sharpe_ratio

    def tracking_error(self,
                       portfolio_returns):

        tracking_error = np.std(portfolio_returns - self.benchmark_returns, ddof=1)

        return tracking_error


class EfficientFrontier():
    def __init__(self,
                 assets_returns,
                 rfr=0.02,
                 n=1000,
                 portfolio_name="Investment Portfolio",
                 daily=False,
                 compounding=True,
                 frequency=252) -> None:

        self.assets_returns = assets_returns
        if daily is True:
            self.rfr=(rfr+1)**(1/252)-1
        else:
            self.rfr=rfr
        self.portfolio_name=portfolio_name
        self.daily=daily
        self.compounding=compounding
        self.frequency=frequency

        np.random.seed(88)
        s = self.assets_returns.columns.shape[0]
        t = self.assets_returns.shape[0]
        assets_returns = self.assets_returns.to_numpy()
        all_weights = np.zeros((n, s))
        returns = np.zeros((n, t))
        self.mean_returns = np.zeros(n)
        self.volatilities = np.zeros(n)
        self.sharpe_ratios = np.zeros(n)

        for i in range(n):
            weights = np.array(np.random.random(s))
            weights = weights/np.sum(weights)
            all_weights[i]=weights

            returns[i]=np.dot(assets_returns, weights)

            if self.daily is False and self.compounding is True:
                self.mean_returns[i] = (1+returns[i]).prod()**(self.frequency/returns[i].shape[0])-1
                self.volatilities[i] = np.nanstd(returns[i], ddof=1)*np.sqrt(self.frequency)
            elif self.daily is False and self.compounding is False:
                self.mean_returns[i] = np.nanmean(returns[i])*self.frequency
                self.volatilities[i] = np.nanstd(returns[i], ddof=1)*np.sqrt(self.frequency)
            elif self.daily is True:
                self.mean_returns[i] = np.nanmean(returns[i])
                self.volatilities[i] = np.nanstd(returns[i], ddof=1)
            else:
                print("Error: mean returns cannot be compounded if daily")

            self.sharpe_ratios[i]=(self.mean_returns[i]-self.rfr)/self.volatilities[i]

        x0=np.full((s), 1/s)
        bounds=tuple(repeat((0,1), s))
        cons_max_sharpe=({"type":"eq", "fun":self._check_weights})
        result_max_sharpe = optimize.minimize(self._negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons_max_sharpe)

        self.best = {"weights":result_max_sharpe.x,
                     "mean_return":self._get_characteristics(result_max_sharpe.x)[0],
                     "volatility":self._get_characteristics(result_max_sharpe.x)[1],
                     "sharpe":-result_max_sharpe.fun}

        self.frontier_y=np.linspace(self.mean_returns.min(),
                                    self.mean_returns.max(),
                                    500)
        self.frontier_x=[]

        # TODO: shorting, leverage
        # TODO: remainder value (make discrete allocations methods)
        for possible_return in self.frontier_y:
            cons_ef = ({'type':'eq', 'fun': self._check_weights},
                       {'type':'eq', 'fun': lambda w: self._get_characteristics(w)[0] - possible_return})
            result_ef = optimize.minimize(self._minimize_volatility,
                                          x0, method='SLSQP',
                                          bounds=bounds, constraints=cons_ef)
            self.frontier_x.append(result_ef['fun'])

        self.frontier_x=np.array(self.frontier_x)

    def plot_efficient_frontier(self,
                                plot_all=False,
                                plot_best=True,
                                plot_assets=False,
                                plot_comparables=False,
                                plot_cal=False,
                                comparables_object=None,
                                comparables_mean_returns=None,
                                comparables_volatilities=None,
                                show=True,
                                save=False):

        fig=plt.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.set_xlabel("Volatility")
        ax1.set_ylabel("Mean Returns")
        ax1.set_title(str(self.portfolio_name+" Efficient Frontier"))
        ax1.plot(self.frontier_x, self.frontier_y,
                 'b-', linewidth=2, label="Efficient Frontier")

        if plot_all is True:
            all_portfolios=ax1.scatter(self.volatilities, self.mean_returns, c=self.sharpe_ratios,
                             cmap='Blues', s=8, marker=".", label="All Portfolios")
            fig.colorbar(all_portfolios, ax=ax1, label="Sharpe Ratio")

        if plot_best is True:
            ax1.scatter(self.best["volatility"], self.best["mean_return"], s=50,
                        c="r", marker="X", zorder=2.6, label="Highest Sharpe Portfolio")

        if plot_assets is True:
            if self.daily is False and self.compounding is True:
                assets_mean_returns = (1+self.assets_returns).prod()**(self.frequency/self.assets_returns.shape[0])-1
                assets_volatilities = self.assets_returns.std()*np.sqrt(self.frequency)
            elif self.daily is False and self.compounding is False:
                assets_mean_returns = np.nanmean(self.assets_returns)*self.frequency
                assets_volatilities = self.assets_returns.std()*np.sqrt(self.frequency)
            elif self.daily is True:
                assets_mean_returns = np.nanmean(self.assets_returns)
                assets_volatilities = self.assets_returns.std()
            else:
                print("Error: mean returns cannot be compounded if daily")
            ax1.scatter(assets_volatilities, assets_mean_returns, s=20,
                        c="g", marker="d", zorder=2.5, label="Portfolio assets")
            for i in range(assets_mean_returns.shape[0]):
                ax1.text(assets_volatilities[i]+0.02*(np.nanmean(assets_volatilities)),
                         assets_mean_returns[i], self.assets_returns.columns[i], size=10)

        if plot_comparables is True:
            if self.daily!=comparables_object.daily or \
               self.compounding!=comparables_object.compounding or \
               self.frequency!=comparables_object.frequency:
                print("Warning: Cannot plot comparables! \n")
                print("Comparables' options (daily, compounding, frequency) don't match efficient frontier options.")
            else:
                ax1.scatter(comparables_volatilities, comparables_mean_returns, s=20,
                            c="orange", marker="x", zorder=2.4, label="Comparable Portfolios")
                for i in range(comparables_mean_returns.shape[0]):
                    ax1.text(1.01*comparables_volatilities[i], comparables_mean_returns[i],
                             comparables_mean_returns.index[i], size=10, color="orange")

        if plot_cal is True:
            ax1.plot(self.frontier_x, self.rfr+self.best["sharpe"]*self.frontier_x,
                     color="black", linewidth=2, label="Capital Allocation Line")

        ax1.legend()
        if save is True:
            plt.savefig(str(self.portfolio_name+"_efficient_frontier.png"), dpi=300)
        if show is True:
            plt.show()

    def _get_characteristics(self,
                             weights):

        portfolio_returns=np.dot(self.assets_returns.to_numpy(), weights)

        if self.daily is False and self.compounding is True:
            mean_return = (1+portfolio_returns).prod()**(self.frequency/portfolio_returns.shape[0])-1
            volatility = np.nanstd(portfolio_returns, ddof=1)*np.sqrt(self.frequency)
        elif self.daily is False and self.compounding is False:
            mean_return = np.nanmean(portfolio_returns)*self.frequency
            volatility = np.nanstd(portfolio_returns, ddof=1)*np.sqrt(self.frequency)
        elif self.daily is True:
            mean_return = np.nanmean(portfolio_returns)
            volatility = np.nanstd(portfolio_returns, ddof=1)
        else:
            print("Error: mean returns cannot be compounded if daily")

        sharpe_ratio=(mean_return-self.rfr)/volatility

        return mean_return, volatility, sharpe_ratio

    def _minimize_volatility(self,
                             weights):

        return self._get_characteristics(weights)[1]

    def _negative_sharpe(self,
                         weights):

        return -1*self._get_characteristics(weights)[2]

    def _check_weights(self,
                       weights):

        return np.sum(weights)-1


class PMPT():
    def __init__(self,
                 benchmark_weights,
                 benchmark_tickers=None,
                 benchmark_data=None,
                 start="1970-01-01",
                 end=str(datetime.now())[0:10]) -> None:

        if benchmark_data is None:
            benchmark_assets_prices=pdr.DataReader(benchmark_tickers, start=start, end=end, data_source="yahoo")["Adj Close"]
            benchmark_assets_returns=benchmark_assets_prices.pct_change()
        else:
            benchmark_assets_prices = pd.read_csv(benchmark_data, index_col=["Date"])    # takes in only the data of kind GetData.save_adj_close_only()
            benchmark_assets_returns = benchmark_assets_prices.pct_change()

        self.benchmark_returns = np.dot(benchmark_assets_returns.to_numpy(), benchmark_weights)
        self.benchmark_returns = np.delete(self.benchmark_returns, [0], axis=0)

    def upside_volatility(self,
                          portfolio_returns,
                          mar=0.03,
                          daily=False,
                          frequency=252):

        mar_daily = (mar + 1)**(1/252)-1

        positive_portfolio_returns = portfolio_returns - mar_daily
        positive_portfolio_returns = positive_portfolio_returns[positive_portfolio_returns>0]
        if daily is False:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1)*frequency
        else:
            upside_volatility = np.std(positive_portfolio_returns, ddof=1)

        return upside_volatility

    def downside_volatility(self,
                            portfolio_returns,
                            mar=0.03,
                            daily=False,
                            frequency=252):

        mar_daily = (mar + 1)**(1/252)-1

        negative_portfolio_returns = portfolio_returns - mar_daily
        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]
        if daily is False:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1)*frequency
        else:
            downside_volatility = np.std(negative_portfolio_returns, ddof=1)

        return downside_volatility

    def volatility_skew(self,
                        portfolio_returns,
                        mar=0.03,
                        daily=False,
                        frequency=252):

        upside = self.upside_volatility(portfolio_returns, mar, daily, frequency)
        downside = self.downside_volatility(portfolio_returns, mar, daily, frequency)
        skew = upside/downside

        return skew

    def omega_excess_return(self,
                            portfolio_returns,
                            mar=0.03,
                            daily=False,
                            frequency=252):

        portfolio_downside_volatility = self.downside_volatility(portfolio_returns, mar, daily, frequency)
        benchmark_downside_volatility = self.downside_volatility(self.benchmark_returns, mar, daily, frequency)

        omega_excess_return = portfolio_returns - 3*portfolio_downside_volatility*benchmark_downside_volatility

        return omega_excess_return

    def upside_potential_ratio(self,
                               portfolio_returns,
                               mar=0.03,
                               daily=False,
                               frequency=252):

        mar_daily = (mar + 1)**(1/252)-1

        downside_volatility = self.downside_volatility(portfolio_returns, mar, daily, frequency)
        upside = portfolio_returns - mar_daily
        upside = upside[upside>0].sum()
        upside_potential_ratio = upside/downside_volatility

        return upside_potential_ratio

    def downside_capm(self,
                      portfolio_returns,
                      mar):

        mar_daily = (mar + 1)**(1/252)-1

        negative_benchmark_returns = self.benchmark_returns - mar_daily
        negative_benchmark_returns = negative_benchmark_returns[negative_benchmark_returns<0]

        negative_portfolio_returns = portfolio_returns - mar_daily
        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]

        model = LinearRegression().fit(negative_benchmark_returns, negative_portfolio_returns)
        downside_alpha = model.intercept_
        downside_beta = model.coef_[0]
        downside_r_squared = model.score(negative_benchmark_returns, negative_portfolio_returns)

        return downside_beta, downside_alpha, downside_r_squared, negative_portfolio_returns, negative_benchmark_returns

    def downside_volatility_ratio(self,
                                  portfolio_returns,
                                  mar=0.03,
                                  daily=False,
                                  frequency=252):

        portfolio_downside_volatility = self.downside_volatility(portfolio_returns, mar, daily, frequency)
        benchmark_downside_volatility = self.downside_volatility(self.benchmark_returns, mar, daily, frequency)

        downside_volatility_ratio = portfolio_downside_volatility/benchmark_downside_volatility

        return downside_volatility_ratio

    def sortino(self,
                portfolio_returns,
                mar=0.03,
                rfr=0.02,
                daily=False,
                compounding=True,
                frequency=252):

        downside_volatility = self.downside_volatility(portfolio_returns, mar, daily, frequency)

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            sortino_ratio = 100*(mean_return - rfr)/downside_volatility
        elif daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            sortino_ratio = 100*(mean_return - rfr)/downside_volatility
        elif daily is True:
            rfr_daily = (rfr + 1)**(1/252)-1
            sortino_ratio = 100*(np.nanmean(portfolio_returns) - rfr_daily)/downside_volatility
        else:
            print("Error: mean returns cannot be compounded if daily")

        return sortino_ratio

    def drawdowns(self, portfolio_returns):
        wealth_index = 1000*(1+portfolio_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        return drawdowns

    def maximum_drawdown(self,
                         portfolio_state,
                         period=1000):

        if period>=portfolio_state.shape[0]:
            period=portfolio_state.shape[0]
            print("Warning! Dataset too small. Period: ", period)

        peak = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = portfolio_state.index.get_loc(peak_index)
        trough = np.min(portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown = trough-peak

        return maximum_drawdown

    def maximum_drawdown_percentage(self,
                                    portfolio_state,
                                    period=1000):

        if period>portfolio_state.shape[0]:
            period=portfolio_state.shape[0]
            print("Warning! Dataset too small. Period: ", period)

        peak = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = portfolio_state.index.get_loc(peak_index)
        trough = np.min(portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown_ratio = (trough-peak)/peak

        return maximum_drawdown_ratio

    def jensen_alpha(self,
                     portfolio_returns,
                     rfr=0.02,
                     daily=False,
                     compounding=True,
                     frequency=252):

        rfr_daily = (rfr + 1)**(1/252)-1

        excess_portfolio_returns = portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(excess_benchmark_returns, excess_portfolio_returns)
        beta = model.coef_[0]

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            mean_benchmark_return=(1+excess_benchmark_returns).prod()**(frequency/excess_benchmark_returns.shape[0])-1
            jensen_alpha = mean_return - rfr - beta*(mean_benchmark_return - rfr)
        if daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            mean_benchmark_return=excess_benchmark_returns.mean()*frequency
            jensen_alpha = mean_return - rfr - beta*(mean_benchmark_return - rfr)
        elif daily is True:
            mean_return=portfolio_returns.mean()
            mean_benchmark_return=excess_benchmark_returns.mean()
            jensen_alpha = mean_return - rfr_daily - beta*(mean_benchmark_return - rfr_daily)
        else:
            print("Error: mean returns cannot be compounded if daily")

        return jensen_alpha

    def treynor(self,
                portfolio_returns,
                rfr=0.02,
                daily=False,
                compounding=True,
                frequency=252):

        rfr_daily = (rfr + 1)**(1/252)-1

        excess_portfolio_returns = portfolio_returns - rfr_daily
        excess_benchmark_returns = self.benchmark_returns - rfr_daily

        model = LinearRegression().fit(excess_benchmark_returns, excess_portfolio_returns)
        beta = model.coef_[0]

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            treynor_ratio = 100*(mean_return-rfr)/beta
        if daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            treynor_ratio = 100*(mean_return-rfr)/beta
        elif daily is True:
            mean_return=portfolio_returns.mean()
            treynor_ratio = 100*(mean_return-rfr_daily)/beta
        else:
            print("Error: mean returns cannot be compounded if daily")

        return treynor_ratio

# TODO do i need to annualize partial moments and kappa?
    def higher_partial_moment(self,
                              portfolio_returns,
                              mar=0.03,
                              moment=3):

        mar_daily = (mar + 1)**(1/252)-1
        days = portfolio_returns.shape[0]

        higher_partial_moment = (1/days)*np.sum(np.power(np.max(portfolio_returns-mar_daily, 0), moment))

        return higher_partial_moment

    def lower_partial_moment(self,
                             portfolio_returns,
                             mar=0.03,
                             moment=3):

        mar_daily = (mar + 1)**(1/252)-1
        days = portfolio_returns.shape[0]

        lower_partial_moment = (1/days)*np.sum(np.power(np.max(mar_daily-portfolio_returns, 0), moment))

        return lower_partial_moment

    def kappa(self,
              portfolio_returns,
              mar=0.03,
              moment=3,
              daily=False,
              compounding=True,
              frequency=252):

        lower_partial_moment = self.lower_partial_moment(portfolio_returns, mar, moment)

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            kappa_ratio = 100*(mean_return - mar)/np.power(lower_partial_moment, (1/moment))
        elif daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            kappa_ratio = 100*(mean_return - mar)/np.power(lower_partial_moment, (1/moment))
        elif daily is True:
            mean_return=portfolio_returns.mean()
            kappa_ratio = 100*(mean_return - mar)/np.power(lower_partial_moment, (1/moment))
        else:
            print("Error: mean returns cannot be compounded if daily")

        return kappa_ratio

    def gain_loss(self,
                  portfolio_returns,
                  mar=0.03,
                  moment=1):

        hpm=self.higher_partial_moment(portfolio_returns, mar, moment)
        lpm=self.lower_partial_moment(portfolio_returns, mar, moment)

        gain_loss_ratio = hpm/lpm

        return gain_loss_ratio
# TODO: the concerned part is until here

    def calmar(self,
               portfolio_returns,
               portfolio_state,
               period=1000,
               rfr=0.02,
               daily=False,
               compounding=True,
               frequency=252):

        if period>=portfolio_state.shape[0]:
            period=portfolio_state.shape[0]
            print("Warning! Dataset too small. Period: ", period)

        maximum_drawdown = self.maximum_drawdown_percentage(portfolio_state, period)

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            calmar_ratio = 100*(mean_return-rfr)/maximum_drawdown
        if daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            calmar_ratio = 100*(mean_return-rfr)/maximum_drawdown
        elif daily is True:
            mean_return=portfolio_returns.mean()
            rfr_daily = (rfr + 1)**(1/252)-1
            calmar_ratio = 100*(mean_return-rfr_daily)/maximum_drawdown

        return calmar_ratio

    def sterling(self,
                 portfolio_returns,
                 rfr=0.02,
                 drawdowns=3,
                 daily=False,
                 compounding=True,
                 frequency=252):


        portfolio_drawdowns = self.drawdowns(portfolio_returns)
        sorted_drawdowns = np.sort(portfolio_drawdowns)
        d_average_drawdown = np.mean(sorted_drawdowns[-drawdowns:])

        if daily is False and compounding is True:
            mean_return=(1+portfolio_returns).prod()**(frequency/portfolio_returns.shape[0])-1
            sterling_ratio = 100*(mean_return - rfr)/np.abs(d_average_drawdown)
        if daily is False and compounding is False:
            mean_return=portfolio_returns.mean()*frequency
            sterling_ratio = 100*(mean_return - rfr)/np.abs(d_average_drawdown)
        elif daily is True:
            mean_return=portfolio_returns.mean()
            rfr_daily = (rfr + 1)**(1/252)-1
            sterling_ratio = 100*(mean_return - rfr_daily)/np.abs(d_average_drawdown)
        else:
            print("Error: mean returns cannot be compounded if daily")

        return sterling_ratio


class Ulcer():
    def __init__(self) -> None:
        pass

    def ulcer(self,
              portfolio_state,
              period=14,
              start=1):

        close = np.empty(period)
        percentage_drawdown = np.empty(period)

        if start==1:
            period_high = np.max(portfolio_state.iloc[-period:]["Whole Portfolio"])
        else:
            period_high = np.max(portfolio_state.iloc[-period-start+1:-start+1]["Whole Portfolio"])

        for i in range(period):
            close[i] = portfolio_state.iloc[-i-start+1]["Whole Portfolio"]
            percentage_drawdown[i] = 100*((close[i] - period_high))/period_high

        ulcer_index = np.sqrt(np.mean(np.square(percentage_drawdown)))

        return ulcer_index

    def martin(self,
               portfolio_state,
               mean_return,
               rfr=0.02,
               period=14):

        ulcer_index = self.ulcer(portfolio_state, period)
        martin_ratio = 100*(mean_return - rfr)/ulcer_index

        return martin_ratio

    def plot_ulcer(self,
                   portfolio_state,
                   period=14,
                   show=True,
                   save=False):

        ulcer_values = pd.DataFrame(columns=["Ulcer Index"], index=portfolio_state.index)
        for i in range(portfolio_state.shape[0]-period):
            ulcer_values.iloc[-i]["Ulcer Index"] = self.ulcer(portfolio_state, period, start=i)

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


class ValueAtRisk():
    def __init__(self) -> None:
        pass

    def analytical_var(self,
                       mean_return,
                       volatility,
                       value,
                       dof,
                       distribution="normal"):

        if distribution=="normal":
            var=stats.norm(mean_return, volatility).cdf(value)
            expected_loss=(stats.norm(mean_return, volatility). \
                          pdf(stats.norm(mean_return, volatility). \
                          ppf((1 - var))) * volatility)/(1 - var) - mean_return
        elif distribution=="t":
            var=stats.t(dof).cdf(value)
            percent_point_function = stats.t(dof).ppf((1 - var))
            expected_loss = -1/(1 - var)*(1-dof)**(-1)*(dof-2 + percent_point_function**2) \
                            *stats.t(dof).pdf(percent_point_function)*volatility-mean_return
        else:
            print("Distribution Unavailable.")

        return var, expected_loss

    def historical_var(self,
                       portfolio_returns,
                       value):

        returns_below_value = portfolio_returns[portfolio_returns<value]
        var=returns_below_value.shape[0]/portfolio_returns.shape[0]

        return var

    def plot_analytical_var(self,
                            mean_return,
                            volatility,
                            value,
                            dof,
                            z=3,
                            distribution="Normal",
                            show=True,
                            save=False):

        x = np.linspace(mean_return-z*volatility, mean_return+z*volatility, 100)

        if distribution=="Normal":
            pdf=stats.norm(mean_return, volatility).pdf(x)
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
                            portfolio_returns,
                            value,
                            number_of_bins=100,
                            show=True,
                            save=False):

        sorted_portfolio_returns = np.sort(portfolio_returns)
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


class Matrices():
    def __init__(self) -> None:
        pass

    def correlation_matrix(self,
                           returns,
                           plot=True,
                           show=True,
                           save=False):

        matrix=returns.corr().round(5)

        if plot is True:
            sns.heatmap(matrix, annot=True, vmin=-1, vmax=1, center=0, cmap="vlag")
            if save is True:
                plt.savefig("correlation_matrix.png", dpi=300)
            if show is True:
                plt.show()

        return matrix

    def covariance_matrix(self,
                          returns,
                          daily=True,
                          frequency=252,
                          plot=True,
                          show=True,
                          save=False):

        if daily is False:
            matrix=returns.cov().round(5)*frequency
        else:
            matrix=returns.cov().round(5)

        if plot is True:
            sns.heatmap(matrix, annot=True, center=0, cmap="vlag")
            if save is True:
                plt.savefig("covariance_matrix.png", dpi=300)
            if show is True:
                plt.show()

        return matrix


class Backtesting():
    def __init__(self) -> None:
        pass


class PortfolioReport():
    def __init__(self) -> None:
        pass


class Helper():
    def __init__(self) -> None:
        pass

    def concatenate_portfolios(self,
                               portfolio_one,
                               portfolio_two):
        """
        Concatenates an array of portfolio returns to an existing array of portfolio returns.
        Accepts array-like objects such as np.ndarray, pd.DataFrame, pd.Series, list etc.

        Args:
            portfolio_one (array-like object): Returns of first portfolio(s).
            portfolio_two (array-like object): Returns of portfolio(s) to be concatenated to the right.

        Returns:
            pd.DataFrame: DataFrame with returns of the given portfolios in respective columns.
        """

        portfolios = pd.concat([pd.DataFrame(portfolio_one), pd.DataFrame(portfolio_two)],
                                axis=1)

        return portfolios

    def periodization(self,
                      given_rate,
                      periods=1/252):
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

        output_rate = (given_rate + 1)**(periods)-1

        return output_rate

    def fill_nan(self,
                 portfolio_returns,
                 method="adjacent",
                 data_object="pandas"):

        if data_object=="numpy":
            portfolio_returns=pd.DataFrame(portfolio_returns)

        if method=="adjacent":
            portfolio_returns.interpolate(method="linear", inplace=True)
        elif method=="column":
            portfolio_returns.fillna(portfolio_returns.mean(), inplace=True)
        else:
            print("Method unsupported.")


class OmegaAnalysis():
    def __init__(self, mar_lower_bound=0, mar_upper_bound=0.2):
        """
        Initiates the object.

        Args:
            mar_lower_bound (int, optional): MAR lower bound for the Omega Curve. Defaults to 0.
            mar_upper_bound (float, optional): MAR upper bound for the Omega Curve. Defaults to 0.2.
        """

        self.mar_array=np.linspace(mar_lower_bound, mar_upper_bound, round(100*(mar_upper_bound-mar_lower_bound)))

    def omega_ratio(self,
                    portfolio_returns,
                    mar=0.03):
        """
        Calculates the Omega Ratio of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame): Dataframe with the daily returns of single or more portfolios.

        Returns:
            pd.Series: Series with Omega Ratios of all portfolios.
        """

        mar_daily = (mar + 1)**np.sqrt(1/252)-1

        excess_returns = portfolio_returns-mar_daily
        winning = excess_returns[excess_returns>0].sum()
        losing = -(excess_returns[excess_returns<=0].sum())

        omega=winning/losing

        return omega

    def omega_curve(self,
                    portfolio_returns,
                    show=True,
                    save=False):
        """
        Plots and/or saves Omega Curve(s) of of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame): Dataframe with the daily returns of single or more portfolios.
            show (bool, optional): Show the plot upon the execution of the code. Defaults to True.
            save (bool, optional): Save the plot on storage. Defaults to False.
        """

        all_values = pd.DataFrame(columns=portfolio_returns.columns)

        for portfolio in portfolio_returns.columns:
            omega_values = list()
            for mar in self.mar_array:
                value = np.round(self.omega_ratio(portfolio_returns[portfolio], mar), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(title="Omega Curve", xlabel="Minimum Acceptable Return (%)", ylabel="Omega Ratio", ylim=(0, 1.5))
        if save is True:
            plt.savefig("omega_curves.png", dpi=300)
        if show is True:
            plt.show()
