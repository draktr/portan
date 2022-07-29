import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self, tickers, weights, start="1970-01-01", end=str(datetime.now())[0:10], data=None, initial_aum=10000, risk_free_rate=0.03):
        self.tickers=tickers
        self.start=start
        self.end=end
        self.data=data # takes in only the data of kind GetData.save_adj_close_only()
        self.initial_aum = initial_aum
        self.risk_free_rate = risk_free_rate # takes in decimals (not percentage)

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

        # funds allocated to each security
        self.funds_allocation = np.multiply(self.initial_aum, weights)

        # number of securities bought at t0
        self.initial_number_of_securities_bought = np.divide(self.funds_allocation, self.prices.iloc[0])

        # absolute (dollar) value of each security in portfolio (i.e. state of the portfolio, not rebalanced)
        self.portfolio_state = np.multiply(self.prices, self.initial_number_of_securities_bought)
        self.portfolio_state["Whole Portfolio"] = self.portfolio_state.sum(axis=1)

        # portfolio returns (numpy array)
        self.portfolio_returns = np.dot(self.securities_returns.to_numpy(), weights)
        self.mean_daily_returns = np.nanmean(self.portfolio_returns)
        self.volatility = np.nanstd(self.portfolio_returns)


    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def returns_with_dates(self):
        # returns pandas dataframe
        returns_with_dates = pd.DataFrame(columns=["Returns"], index=self.securities_returns.index)
        returns_with_dates["Returns"] = self.portfolio_returns
        return returns_with_dates

    def portfolio_cumulative_returns(self):
        portfolio_returns = self.returns_with_dates()
        portfolio_cumulative_returns = (portfolio_returns["Returns"] + 1).cumprod()
        return portfolio_cumulative_returns

    def plot_portfolio_returns(self):
        portfolio_returns = self.returns_with_dates()

        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns["Returns"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily Returns")
        ax1.set_title("Portfolio Daily Returns")
        mpl.show()

    def plot_portfolio_returns_distribution(self):
        portfolio_returns = self.returns_with_dates()

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_returns.plot.hist(bins = 90)
        ax1.set_xlabel("Daily Returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Portfolio Returns Distribution")
        mpl.show()

    def plot_portfolio_cumulative_returns(self):
        portfolio_cumulative_returns = self.portfolio_cumulative_returns()

        fig = mpl.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        portfolio_cumulative_returns.plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("Portfolio Cumulative Returns")
        mpl.show()

    def excess_return(self, minimum_acceptable_return=0.03):
        excess_return = self.portfolio_returns - minimum_acceptable_return
        return excess_return

    def benchmark(self, benchmark_tickers, benchmark_weights, data=None):
        if data is None:
            benchmark_securities_prices = pd.DataFrame(columns=benchmark_tickers)
            benchmark_securities_returns = pd.DataFrame(columns=benchmark_tickers)

            for ticker in benchmark_tickers:
                price_current = yf.download(ticker, start=self.start, end=self.end)
                benchmark_securities_prices[ticker] = price_current["Adj Close"]
                benchmark_securities_returns[ticker] = price_current["Adj Close"].pct_change()

            benchmark_returns = np.dot(benchmark_securities_returns.to_numpy(), benchmark_weights)
        else:
            benchmark_securities_prices = pd.read_csv(data, index_col=["Date"]) # takes in only the data of kind GetData.save_adj_close_only()
            benchmark_securities_returns = pd.DataFrame(columns=benchmark_tickers, index=benchmark_securities_prices.index)
            benchmark_securities_returns = benchmark_securities_prices.pct_change()

            benchmark_returns = np.dot(benchmark_securities_returns.to_numpy(), benchmark_weights)

        return benchmark_returns

    def capm(self, benchmark_tickers, benchmark_weights, data=None):
        excess_portfolio_returns = self.portfolio_returns - self.risk_free_rate
        excess_benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        excess_benchmark_returns = excess_benchmark_returns - self.risk_free_rate

        model = LinearRegression().fit(excess_benchmark_returns, excess_portfolio_returns)
        alpha = model.intercept_
        beta = model.coef_
        r_squared = model.score(excess_benchmark_returns, excess_portfolio_returns)

        return alpha, beta, r_squared

    def plot_capm(self, benchmark_tickers, benchmark_weights, data=None, show=True, save=False):

        excess_portfolio_returns = self.portfolio_returns - self.risk_free_rate
        excess_benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        excess_benchmark_returns = excess_benchmark_returns - self.risk_free_rate

        beta, alpha = np.polyfit(excess_benchmark_returns, excess_portfolio_returns, 1)

        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.scatter(excess_benchmark_returns, excess_portfolio_returns)
        ax1.plot(excess_benchmark_returns, alpha+beta*excess_benchmark_returns)
        ax1.legend(labels=[r'$\alpha$', r'$\beta$']) #TODO check if this works
        ax1.set_xlabel("Benchmark Excess Returns")
        ax1.set_ylabel("Portfolio Excess Returns")
        ax1.set_title("Portfolio Excess Returns Against Benchmark")
        if save is True:
            mpl.savefig("capm.png", dpi=300)
        if show is True:
            mpl.show()

    def sharpe(self):
        sharpe_ratio = 100*(self.mean_daily_returns - self.risk_free_rate)/self.volatility
        return sharpe_ratio

    def upside_volatility(self):
        positive_portfolio_returns = self.portfolio_returns - self.risk_free_rate
        positive_portfolio_returns = positive_portfolio_returns[positive_portfolio_returns>0]
        upside_volatility = np.std(positive_portfolio_returns)
        return upside_volatility

    def downside_volatility(self, returns=None): #? use MAR instead of RFR?
        if returns is None:
            negative_portfolio_returns = self.portfolio_returns - self.risk_free_rate
        else:
            negative_portfolio_returns = returns - self.risk_free_rate

        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]
        downside_volatility = np.std(negative_portfolio_returns)
        return downside_volatility

    def volatility_skewness(self):
        upside = self.upside_volatility()
        downside = self.downside_volatility()
        skewness = upside/downside
        return skewness

    def omega_excess_return(self, benchmark_tickers, benchmark_weights, data=None):
        portfolio_downside_volatility = self.downside_volatility()
        benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        benchmark_downside_volatility = self.downside_volatility(benchmark_returns)

        omega_excess_return = self.portfolio_returns - 3*portfolio_downside_volatility*benchmark_downside_volatility
        return omega_excess_return

    def upside_potential_ratio(self):
        downside_volatility = self.downside_volatility()
        upside = self.portfolio_returns - self.risk_free_rate
        upside = upside[upside>0].sum()
        upside_potential_ratio = upside/downside_volatility
        return upside_potential_ratio

    def downside_capm(self, benchmark_tickers, benchmark_weights, data=None, minimum_acceptable_return=0.03):
        negative_benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)
        negative_benchmark_returns = negative_benchmark_returns - minimum_acceptable_return
        negative_benchmark_returns = negative_benchmark_returns[negative_benchmark_returns<0]

        negative_portfolio_returns = self.portfolio_returns - minimum_acceptable_return
        negative_portfolio_returns = negative_portfolio_returns[negative_portfolio_returns<0]

        model = LinearRegression().fit(negative_benchmark_returns, negative_portfolio_returns)
        downside_alpha = model.intercept_
        downside_beta = model.coef_
        downside_r_squared = model.score(negative_benchmark_returns, negative_portfolio_returns)

        return downside_beta, downside_alpha, downside_r_squared

    def downside_volatility_ratio(self, benchmark_tickers, benchmark_weights, data=None):
        benchmark_returns = self.benchmark(benchmark_tickers, benchmark_weights, data)

        portfolio_downside_volatility = self.downside_volatility()
        benchmark_downside_volatility = self.downside_volatility(benchmark_returns)

        downside_volatility_ratio = portfolio_downside_volatility/benchmark_downside_volatility
        return downside_volatility_ratio

    def sortino(self):
        downside_volatility = self.downside_volatility()
        sortino_ratio = 100*(self.mean_daily_returns - self.risk_free_rate)/downside_volatility
        return sortino_ratio

    def maximum_drawdown(self, period=365):
        peak = np.max(self.portfolio_state.iloc[-period:]["Whole Portfolio"])
        peak_index = self.portfolio_state["Whole Portfolio"].idxmax()
        peak_index_int = self.portfolio_state.index.get_loc(peak_index)
        trough = np.min(self.portfolio_state.iloc[-peak_index_int:]["Whole Portfolio"])

        maximum_drawdown = (trough-peak)/peak
        return maximum_drawdown

    def ulcer(self, period=14, start=1):
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

    def plot_ulcer(self, period=14):
        ulcer_values = pd.DataFrame(columns=["Ulcer Index"], index=self.portfolio_state.index)
        for i in range(len(self.portfolio_state)-period):
            ulcer_values.iloc[-i]["Ulcer Index"] = self.ulcer(period, start=i)

        fig=mpl.figure()
        ax1=fig.add_axes([0.1,0.1,0.8,0.8])
        ulcer_values["Ulcer Index"].plot()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Ulcer Index")
        ax1.set_title("Portfolio Ulcer Index")
        mpl.show()

    def martin(self, period=14):
        ulcer_index = self.ulcer(period)
        sharpe_ratio = 100*(self.mean_daily_returns - self.risk_free_rate)/ulcer_index
        return sharpe_ratio

    def omega_ratio(self, annual_threshold=0.03):
        daily_threshold = (annual_threshold + 1)**np.sqrt(1/252)-1
        returns = self.returns_with_dates()

        excess_returns = returns-daily_threshold
        winning = excess_returns[excess_returns>0].sum()
        losing = -(excess_returns[excess_returns<=0].sum())

        omega=winning/losing
        return omega

# TODO: same analytics for each security in the portfolio separately
# TODO: other ratios
# TODO: ulcer index adn other measurements of pain
# TODO: do portfolio metrics in one method
# TODO: separate one for checking which weigths
# TODO: asset covariance or correlation matrix
# TODO: excess stuff for alpha and beta analysis
# TODO: mpl to plt
# TODO: replace threshold with MAR, in omega_analysis as well
# ? annualizing ratios?
# period=len(data) | x*np.sqrt(period)

# cloud providers, chip names (SOXX), cyber security
