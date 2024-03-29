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


def test_sterling(portfolio):
    m = portfolio.sterling()

    assert np.abs(m - 0.4213370777156648) < 0.01


def test_calmar(portfolio):
    m = portfolio.calmar()

    assert np.abs(m - 0.15656131465177014) < 0.01


def test_gain_loss(portfolio):
    m = portfolio.gain_loss()

    assert np.abs(m - 1.08548878) < 0.01


def test_kappa(portfolio):
    m = portfolio.kappa(annual=False)

    assert np.abs(m - 0.01841138) < 0.01


def test_lpm(portfolio):
    m = portfolio.lpm()

    assert np.abs(m - 1.0621206e-06) < 0.01


def test_hpm(portfolio):
    m = portfolio.hpm()

    assert np.abs(m - 1.0788034e-06) < 0.01


def test_treynor(portfolio):
    m = portfolio.treynor()

    assert np.abs(m - 0.0179078) < 0.01


def test_jensen_alpha(portfolio):
    m = portfolio.jensen_alpha()

    assert np.abs(m - -0.07671839837596321) < 0.01


def test_sortino(portfolio):
    m = portfolio.sortino(annual=False)

    assert np.abs(m - 0.02646895) < 0.01


def test_omega_excess_return(portfolio):
    m = portfolio.omega_excess_return()

    assert np.abs(m - 0.05446476) < 0.01


def test_volatility_skewness(portfolio):
    m = portfolio.volatility_skewness()

    assert np.abs(m - 1.02637) < 0.01


def test_information_ratio(portfolio):
    m = portfolio.information_ratio()

    assert np.abs(m - -0.27850399828383127) < 0.01


def test_tracking_error(portfolio):
    m = portfolio.tracking_error()

    assert np.abs(m - 0.11847556606072403) < 0.01
