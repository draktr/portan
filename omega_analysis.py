import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OmegaAnalysis():
    def __init__(self, returns, mar_lower_bound=0, mar_upper_bound=0.2):
        """
        Initiates the object.

        Args:
            returns (pd.DataFrame): Dataframe with the daily returns of different portfolios.
            mar_lower_bound (int, optional): MAR lower bound for the Omega Curve. Defaults to 0.
            mar_upper_bound (float, optional): MAR upper bound for the Omega Curve. Defaults to 0.2.
        """

        self.returns=returns
        self.mar_array=np.linspace(mar_lower_bound, mar_upper_bound, round(100*(mar_upper_bound-mar_lower_bound)))

    def omega_ratio(self,
                    portfolio_returns=None,
                    mar=0.03):
        """
        Calculates the Omega Ratio of one or more portfolios.

        Args:
            portfolio_returns (pd.DataFrame, optional): Dataframe with the daily returns of single or more portfolios. Defaults to None.

        Returns:
            pd.Series: Series with Omega Ratios of all portfolios.
        """

        mar_daily = (mar + 1)**np.sqrt(1/252)-1

        if portfolio_returns is None:
            excess_returns = self.returns-mar_daily
            winning = excess_returns[excess_returns>0].sum()
            losing = -(excess_returns[excess_returns<=0].sum())

            omega=winning/losing
        else:
            excess_returns = portfolio_returns-mar_daily
            winning = excess_returns[excess_returns>0].sum()
            losing = -(excess_returns[excess_returns<=0].sum())

            omega=winning/losing

        return omega

    def omega_curve(self,
                    show=True,
                    save=False):
        """
        Plots and/or saves Omega Curve(s) of of one or more portfolios.

        Args:
            show (bool, optional): Show the plot upon the execution of the code. Defaults to True.
            save (bool, optional): Save the plot on storage. Defaults to False.
        """

        all_values = pd.DataFrame(columns=self.returns.columns)

        for portfolio in self.returns.columns:
            omega_values = list()
            for mar in self.mar_array:
                value = np.round(self.omega_ratio(self.returns[portfolio], mar), 5)
                omega_values.append(value)
            all_values[portfolio] = omega_values

        all_values.plot(title="Omega Curve", xlabel="Minimum Acceptable Return (%)", ylabel="Omega Ratio", ylim=(0, 1.5))
        if save is True:
            plt.savefig("omega_curves.png", dpi=300)
        if show is True:
            plt.show()
