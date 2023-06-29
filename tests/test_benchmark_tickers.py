import pytest
import numpy as np
from portfolio_analytics.portfolio_analytics import PortfolioAnalytics


@pytest.fixture
def portfolio():
    portfolio = PortfolioAnalytics(
        tickers=["XOM", "GOOG"],
        weights=[0.3, 0.7],
        benchmark_tickers=["ITOT", "IEF"],
        benchmark_weights=[0.6, 0.4],
        start="2012-01-01",
        end="2020-01-01",
    )

    return portfolio


def test_tickers_geometric_mean(portfolio):
    assert np.abs(portfolio.geometric_mean - 0.0661235) < 0.01


def test_tickers_information_ratio(portfolio):
    assert np.abs(portfolio.information_ratio() - -0.06629389) < 0.01
