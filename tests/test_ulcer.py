import pytest
import numpy as np
from portan import Analytics
from portan import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2020-01-01").data[
        "Close"
    ]
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2020-01-01"
    ).data["Close"]

    portfolio = Analytics(
        prices=data,
        weights=[0.7, 0.3],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_ulcer(portfolio):
    m = portfolio.ulcer()

    assert np.all(np.abs(m - 6.805084369330141) < 0.01)


def test_martin(portfolio):
    m = portfolio.martin()

    assert np.abs(m - 0.005310959760627298) < 0.01
