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
        weights=[0.3, 0.7],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_downside_risk(portfolio):
    m = portfolio.downside_risk()

    assert np.abs(m - 0.007101652) < 0.01


def test_downside_potential(portfolio):
    m = portfolio.downside_potential()

    assert np.abs(m - 0.003618607) < 0.01


def test_downside_variance(portfolio):
    m = portfolio.downside_variance()

    assert np.abs(m - 5.04467e-05) < 0.01


def test_upside_risk(portfolio):
    m = portfolio.upside_risk()

    assert np.abs(m - 0.007194679) < 0.01


def test_upside_potential(portfolio):
    m = portfolio.upside_potential()

    assert np.abs(m - 0.0038083) < 0.01


def test_upside_variance(portfolio):
    m = portfolio.upside_variance()

    assert np.abs(m - 5.1764147e-05) < 0.01
