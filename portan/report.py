from portan import Analytics


class Report:
    def __init__(self) -> None:
        pass


def tearsheet(self, rfr=0.02, mar=0.03):
    parameters = {"rfr": rfr, "mar": mar}
    self.tickers
    self.weights
    self.plot_piechart()

    self.geometric_mean
    self.annual_volatility

    Modern().sharpe()
    Modern().plot_capm()

    PostModern().downside_capm()
    PostModern().sortino()
    PostModern().maximum_drawdown()
    PostModern().jensen_alpha()
    PostModern().treynor()
    PostModern().kappa()
    PostModern().calmar()
    PostModern().sterling()

    Ulcer.martin()
    Ulcer.plot_ulcer()

    ValueAtRisk().plot_analytical_var()
    ValueAtRisk().plot_historical_var()

    Matrices().plot_correlation()

    Omega().omega_ratio()
    Omega().plot_omega_curve()
