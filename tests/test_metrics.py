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


def test_herfindahl_index(portfolio):
    m = portfolio.herfindahl_index()

    assert np.abs(m - 0.1570088) < 0.001


def test_appraisal(portfolio):
    m = portfolio.appraisal()

    assert np.abs(m - -0.701247436948423) < 0.001


def test_burke(portfolio):
    m = portfolio.burke()

    assert np.abs(m - 0.0109257) < 0.001


def test_hurst_index(portfolio):
    m = portfolio.hurst_index()

    assert np.abs(m - 0.3225667) < 0.01


def test_bernardo_ledoit(portfolio):
    m = portfolio.bernardo_ledoit()

    assert np.abs(m - 1.085735) < 0.01


def test_skewness_kurtosis_ratio(portfolio):
    m = portfolio.skewness_kurtosis_ratio()

    assert np.abs(m - -0.01333281) < 0.01


def test_d(portfolio):
    m = portfolio.d()

    assert np.abs(m - 0.8859909) < 0.01


def test_kelly_criterion(portfolio):
    m = portfolio.kelly_criterion()

    assert np.abs(m - 1.8399636985826249) < 0.01


def test_modigliani(portfolio):
    m = portfolio.modigliani(annual=False, compounding=False)

    assert np.abs(m - 0.0001749807) < 0.01


def test_fama_beta(portfolio):
    m = portfolio.fama_beta()

    assert np.abs(m - 2.2315815270153005) < 0.01


def test_diversification(portfolio):
    m = portfolio.diversification()

    assert np.abs(m - 0.041425953987263654) < 0.01


def test_net_selectivity(portfolio):
    m = portfolio.net_selectivity()

    assert np.abs(m - -0.1111744) < 0.01


def test_downside_frequency(portfolio):
    m = portfolio.downside_frequency()

    assert np.abs(m - 0.4967678) < 0.01


def test_upside_frequency(portfolio):
    m = portfolio.upside_frequency()

    assert np.abs(m - 0.5032322) < 0.01
