import pytest
import numpy as np
from portan import Analytics
from portan import GetData


@pytest.fixture
def portfolio():
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2020-01-01"
    ).data["Close"]

    portfolio = Analytics(
        tickers=["XOM", "GOOG"],
        weights=[0.7, 0.3],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
        start="2012-01-01",
        end="2020-01-01",
    )

    return portfolio


def test_tickers_geometric_mean(portfolio):
    assert np.abs(portfolio.geometric_mean - 0.0661235) < 0.01


def test_tickers_information_ratio(portfolio):
    assert np.abs(portfolio.information_ratio() - -0.27850399828383127) < 0.01
