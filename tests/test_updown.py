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


def test_up_capture(portfolio):
    m = portfolio.up_capture()

    assert np.abs(m - 1.439765554985202) < 0.01


def test_down_capture(portfolio):
    m = portfolio.down_capture()

    assert np.abs(m - 1.6122548053867927) < 0.01


def test_up_number(portfolio):
    m = portfolio.up_number()

    assert np.abs(m - 0.7049469964664311) < 0.01


def test_down_number(portfolio):
    m = portfolio.down_number()

    assert np.abs(m - 0.7417519908987485) < 0.01


def test_up_percentage(portfolio):
    m = portfolio.up_percentage()

    assert np.abs(m - 0.573321554770318) < 0.01


def test_down_percentage(portfolio):
    m = portfolio.down_percentage()

    assert np.abs(m - 0.3856655290102389) < 0.01
