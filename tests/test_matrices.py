import pytest
import numpy as np
from portan import Analytics
from portan import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2012-02-01").data[
        "Close"
    ]
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2012-02-01"
    ).data["Close"]

    portfolio = Analytics(
        prices=data,
        weights=[0.3, 0.7],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_cor(portfolio):
    cor = portfolio.correlation()

    assert np.all(np.abs(cor - np.array([[1.0, -0.26931], [-0.26931, 1.0]])) < 0.01)


def test_cor(portfolio):
    cov = portfolio.covariance()

    assert np.all(
        np.abs(cov - np.array([[5.3e-04, -5.0e-05], [-5.0e-05, 5.0e-05]])) < 0.01
    )
