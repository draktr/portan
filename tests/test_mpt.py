import pytest
import numpy as np
from portfolio_analytics.portfolio_analytics import PortfolioAnalytics
from portfolio_analytics.get_data import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2020-01-01").data[
        "Close"
    ]
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2020-01-01"
    ).data["Close"]

    portfolio = PortfolioAnalytics(
        prices=data,
        weights=[0.3, 0.7],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_excess_return(portfolio):
    m = portfolio.excess_return()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_sharpe(portfolio):
    m = portfolio.sharpe()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_capm(portfolio):
    m = portfolio.capm()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_capm_return(portfolio):
    m = portfolio.capm_return()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_ewm(portfolio):
    m = portfolio.ewm()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_distribution_test(portfolio):
    m = portfolio.v()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_net_return(portfolio):
    m = portfolio.net_return()

    assert np.abs(m - -1.5833351818499246) < 0.01


def test_excess_return_above_mar(portfolio):
    m = portfolio.excess_return_above_mar()

    assert np.abs(m - -1.5833351818499246) < 0.01
