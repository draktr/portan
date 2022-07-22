import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import yfinance as yf


class PortfolioAnalytics():
    def __init__(self, tickers, weights, start, end, initial_aum=10000, risk_free_rate=0.01):
        self.tickers=tickers
        self.start=start
        self.end=end
        # ? self.weights = weights
        self.initial_aum = initial_aum
        self.risk_free_rate = risk_free_rate

        self.securities_returns = pd.DataFrame(columns=self.tickers)
        self.prices = pd.DataFrame(columns=self.tickers)
        for ticker in self.tickers:
            price_current = yf.download(ticker, start=self.start, end=self.end)
            self.prices[ticker] = price_current["Adj Close"]
            self.securities_returns[ticker] = price_current["Adj Close"].pct_change()

        # funds allocated to each security
        self.funds_allocation = np.multiply(self.initial_aum, weights)

        # number of securities bought at t0
        self.initial_number_of_securities_bought = np.divide(self.funds_allocation, self.prices.iloc[0])

        # absolute (dollar) value of each security in portfolio (i.e. state of the portfolio, not rebalanced)
        self.portfolio_state = np.multiply(self.prices, self.initial_number_of_securities_bought)
        self.portfolio_state["Whole Portfolio"] = self.portfolio_state.sum(axis=1)

        # portfolio returns (numpy array)
        self.portfolio_returns = np.dot(self.securities_returns.to_numpy(), weights)
        #TODO: if-else about importing .csv data, periods etc

    def list_securities(self):
        for ticker in self.tickers:
            security = yf.Ticker(ticker)
            print(security.info["longName"])

    def returns_with_dates(self):
        # returns pandas dataframe
        returns_with_dates = pd.DataFrame(columns=["Returns"], index=self.securities_returns.index)
        returns_with_dates["Returns"] = self.portfolio_returns
        return returns_with_dates

    def portfolio_daily_volatility(self):
        volatility = np.std(self.portfolio_returns)
        return volatility

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

    def average_daily_return(self):
        returns = self.returns_with_dates()
        average_return = returns.mean()
        return average_return

    def sharpe(self):
        mean_daily_returns = np.mean(self.portfolio_returns)
        volatility = np.std(self.portfolio_returns)
        sharpe_ratio = (100*mean_daily_returns - self.risk_free_rate)/volatility
        return sharpe_ratio

    def downside_deviation(self):
        portfolio_returns = self.portfolio_returns - self.risk_free_rate
        portfolio_returns = portfolio_returns[portfolio_returns<0]
        downside_deviation = np.std(portfolio_returns)
        return downside_deviation

    def sortino(self):
        mean_daily_returns = np.mean(self.portfolio_returns)
        downside_volatility = self.downside_deviation()
        sortino_ratio = (100*mean_daily_returns - self.risk_free_rate)/downside_volatility
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
        mean_daily_returns = np.mean(self.portfolio_returns)
        ulcer_index = self.ulcer(period)
        sharpe_ratio = (100*mean_daily_returns - self.risk_free_rate)/ulcer_index
        return sharpe_ratio

# TODO: same analytics for each security in the portfolio separately
# TODO: other ratios
# TODO: ulcer index adn other measurements of pain
# TODO: do portfolio metrics in one method
# TODO: separate one for checking which weigths
# TODO: asset covariance or correlation matrix
# TODO figure out 100* percentage stuff for the ratios
# TODO: look into martin, sharepe etc bc nans

# TODO: figure out what else can be put in the __init__() (mean_daily_return etc)
# TODO: market report (indices, yield curve etc)

# cloud providers, chip names (SOXX), cyber security
