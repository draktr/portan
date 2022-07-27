import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

class OmegaAnalysis():
    def __init__(self, returns):
        self.returns=returns # Dataframe with the daily returns of different portfolios
                             # Each col is a return time series of a particular portfolio to be compared
        self.thresholds = np.linspace(0, 1, 100)

    def omega_ratio(self, portfolio_returns=None, annual_threshold=0.03):
        daily_threshold = (annual_threshold + 1)**np.sqrt(1/252)-1

        if portfolio_returns is None:
            excess_returns = self.returns-daily_threshold
            winning = excess_returns[excess_returns>0].sum()
            losing = -(excess_returns[excess_returns<=0].sum())

            omega=winning/losing
        else:
            excess_returns = portfolio_returns-daily_threshold
            winning = excess_returns[excess_returns>0].sum()
            losing = -(excess_returns[excess_returns<=0].sum())

            omega=winning/losing
        return omega

    def omega_curve(self):
        all_values = pd.DataFrame(columns=self.returns.columns)
        for portfolio in self.returns.columns:
            omega_values = list()
            for threshold in self.thresholds:
                value = np.round(self.omega_ratio(self.returns[portfolio], threshold), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(title="Omega Curve", xlabel="Minimum Acceptable Return (%)", ylabel="Omega Ratio", ylim=(0, 1.5))
        mpl.show()
