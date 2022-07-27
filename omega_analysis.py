import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

class OmegaAnalysis():
    def __init__(self, returns, threshold_lower=0, threshold_upper=0.2):
        """
        Initiates the object.

        Args:
            returns (pd.DataFrame): Dataframe with the daily returns of different portfolios.
            threshold_lower (int, optional): Threshold lower bound for the Omega Curve. Defaults to 0.
            threshold_upper (float, optional): Threshold upper bound for the Omega Curve. Defaults to 0.2.
        """

        self.returns=returns
        self.thresholds=np.linspace(threshold_lower, threshold_upper, round(100*(threshold_upper-threshold_lower)))

    def omega_ratio(self, portfolio_returns=None, annual_threshold=0.03):
        """
        Calculates the Omega Ratio of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame, optional): Dataframe with the daily returns of single or more portfolios. Defaults to None.
            annual_threshold (float, optional): Minimum Acceptable Return (decimal). Defaults to 0.03.

        Returns:
            pd.Series: Series with Omega Ratios of all portfolios.
        """

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

    def omega_curve(self, show=True, save=False):
        """
        Plots and/or saves Omega Curve(s) of of one or more portfolios.

        Args:
            show (bool, optional): Show the plot upon the execution of the code. Defaults to True.
            save (bool, optional): Save the plot on storage. Defaults to False.
        """

        all_values = pd.DataFrame(columns=self.returns.columns)

        for portfolio in self.returns.columns:
            omega_values = list()
            for threshold in self.thresholds:
                value = np.round(self.omega_ratio(self.returns[portfolio], threshold), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(title="Omega Curve", xlabel="Minimum Acceptable Return (%)", ylabel="Omega Ratio", ylim=(0, 1.5))
        if save is True:
            mpl.savefig("omega_curves.png", dpi=300)
        if show is True:
            mpl.show()
