import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy import optimize
from itertools import repeat
import pypfopt as ppo

#? how acceptable is it to update object properties (self.) with methods
# ? post-processing

class AssetAllocation():
    def __init__(self) -> None:
        pass


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
        # TODO: add minimum volatility
        # TODO: efficeint return, risk
        # TODO: add max weightsed sharpe ratio
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
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Mean Returns")
        ax.set_title(str(self.portfolio_name+" Efficient Frontier"))
        ax.plot(self.frontier_x, self.frontier_y,
                 'b-', linewidth=2, label="Efficient Frontier")

        if plot_all is True:
            all_portfolios=ax.scatter(self.volatilities, self.mean_returns, c=self.sharpe_ratios,
                             cmap='Blues', s=8, marker=".", label="All Portfolios")
            fig.colorbar(all_portfolios, ax=ax, label="Sharpe Ratio")

        if plot_best is True:
            ax.scatter(self.best["volatility"], self.best["mean_return"], s=50,
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
            ax.scatter(assets_volatilities, assets_mean_returns, s=20,
                        c="g", marker="d", zorder=2.5, label="Portfolio assets")
            for i in range(assets_mean_returns.shape[0]):
                ax.text(assets_volatilities[i]+0.02*(np.nanmean(assets_volatilities)),
                         assets_mean_returns[i], self.assets_returns.columns[i], size=10)

        if plot_comparables is True:
            if self.daily!=comparables_object.daily or \
               self.compounding!=comparables_object.compounding or \
               self.frequency!=comparables_object.frequency:
                print("Warning: Cannot plot comparables! \n")
                print("Comparables' options (daily, compounding, frequency) don't match efficient frontier options.")
            else:
                ax.scatter(comparables_volatilities, comparables_mean_returns, s=20,
                            c="orange", marker="x", zorder=2.4, label="Comparable Portfolios")
                for i in range(comparables_mean_returns.shape[0]):
                    ax.text(1.01*comparables_volatilities[i], comparables_mean_returns[i],
                             comparables_mean_returns.index[i], size=10, color="orange")

        if plot_cal is True:
            ax.plot(self.frontier_x, self.rfr+self.best["sharpe"]*self.frontier_x,
                     color="black", linewidth=2, label="Capital Allocation Line")

        ax.legend()
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

#TODO: difference in risk models between using cov matrix and volatilities

class PyPortfolioOptAllocators():
    def __init__(self,
                 prices,
                 weight_bounds=(0, 1)) -> None:
        self.prices=prices
        self.weight_bounds=weight_bounds

        self.returns=ppo.expected_returns.returns_from_prices(self.prices)
        self.mean_returns = ppo.expected_returns.mean_historical_return(self.prices)
        self.cov_matrix = ppo.risk_models.sample_cov(self.prices)


    def sharpe_maximizer(self,
                         target_return,
                         market_neutral,
                         gamma,
                         efficient_return=False,
                         use_objective=False):
        mu = ppo.expected_returns.mean_historical_return(self.prices)
        S = ppo.risk_models.sample_cov(self.prices)

        ef = ppo.EfficientFrontier(mu, S, self.weight_bounds)
        cleaned_weights = ef.clean_weights()

        if efficient_return is True:    #TODO: split efficient return and others
                                        #TODO: into other methods
            ef.efficient_return(target_return, market_neutral)

        if use_objective is True:
            ef.add_objective(ppo.objective_functions.L2_reg, gamma)

        expected_performance = ef.portfolio_performance(verbose=True)

        return cleaned_weights, expected_performance

    def discrete_allocation(self,
                            weights,
                            aum):
        latest_prices = pdr.get_quote_yahoo(self.prices.columns)["regularMarketPrice"]
        da = ppo.DiscreteAllocation(weights, latest_prices, total_portfolio_value=aum)
        allocation, leftover = da.greedy_portfolio()

        return allocation, leftover

#TODO: do this procedural way or all in one function so that i gett all the possible allocations

    def black_litterman(self):
        S = ppo.risk_models.sample_cov(self.prices)
        viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
        bl = ppo.BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
        rets = bl.bl_returns()

        ef = EfficientFrontier(rets, S, self.weight_bounds)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_performance = ef.portfolio_performance(verbose=True)

        return cleaned_weights, expected_performance

    def mean_semivariance(self):
        es = ppo.EfficientSemivariance(self.mean_returns, self.returns)
        es.efficient_return(target_return=0.20)
        weights = es.clean_weights()
        expected_performance = es.portfolio_performance(verbose=True)

        return weights, expected_performance

    def efficient_cvar(self):
        ecvar = ppo.EfficientCVaR(self.mean_returns, self.returns)
        ecvar.min_cvar(target_return=0.20)
        weights = ecvar.clean_weights()
        expected_performance = ecvar.portfolio_performance(verbose=True)

        return weights, expected_performance

    def efficient_cdar(self):
        ecdar = ppo.EfficientCDaR(self.mean_returns, self.returns)
        ecdar.min_cdar()
        weights = ecdar.clean_weights()
        expected_performance = ecdar.portfolio_performance(verbose=True)


    def hrp(self):

        hrp = ppo.HRPOpt(self.mean_returns, self.cov_matrix)
        hrp.optimize()
        weights=hrp.clean_weights()
        expected_performance = hrp.portfolio_performance(verbose=True)

        return weights, expected_performance

    def cla(self):

        cla = ppo.CLA(self.mean_returns, self.cov_matrix)
        cla.efficient_frontier()

        weights = cla.clean_weights()
        expected_performance = cla.portfolio_performance(verbose=True)

        return weights, expected_performance




#TODO: pytest, black, BiBTex, inheritance, polimorphism
#TODO: why cov matrix?
#TODO: explore base optimizer
#TODO: full packages suite



