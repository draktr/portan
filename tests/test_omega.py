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


def test_omega_ratio(portfolio):
    m = portfolio.omega_ratio()

    assert np.abs(m - 1.051946) < 0.01


def test_omega_sharpe_ratio(portfolio):
    m = portfolio.omega_sharpe_ratio()

    assert np.abs(m - 0.05194632) < 0.01
