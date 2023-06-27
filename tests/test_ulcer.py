import pytest
import numpy as np
from portfolio_analytics.portfolio_analytics import PortfolioAnalytics
from portfolio_analytics.get_data import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2012-03-01").data[
        "Close"
    ]
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2012-03-01"
    ).data["Close"]

    portfolio = PortfolioAnalytics(
        prices=data,
        weights=[0.3, 0.7],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_ulcer(portfolio):
    m = portfolio.ulcer()

    assert np.all(np.abs(m - np.array()) < 0.01)


def test_martin(portfolio):
    m = portfolio.martin()

    assert np.abs(m - -0.0152203607354951) < 0.01


portfolio.ulcer()
portfolio.martin()
