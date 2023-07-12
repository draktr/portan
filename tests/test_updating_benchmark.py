import pytest
import numpy as np
from portan import Analytics
from portan import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2020-01-01").data[
        "Close"
    ]

    portfolio = Analytics(
        prices=data,
        weights=[0.7, 0.3],
    )

    return portfolio


def test_updating_prices(portfolio):
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2020-01-01"
    ).data["Close"]
    ret = portfolio.capm_return(
        benchmark={
            "benchmark_prices": benchmark,
            "benchmark_weights": [0.6, 0.4],
        }
    )
    ratio = portfolio.information_ratio()

    assert (np.abs(ret - 0.14285995167253468) < 0.01) & (
        np.abs(ratio - -0.27850399828383127) < 0.01
    )


def test_updating_tickers(portfolio):
    ret = portfolio.capm_return(
        benchmark={
            "benchmark_tickers": ["ITOT", "IEF"],
            "benchmark_weights": [0.6, 0.4],
        }
    )
    ratio = portfolio.information_ratio()

    assert (np.abs(ret - 0.14285995167253468) < 0.01) & (
        np.abs(ratio - -0.27850399828383127) < 0.01
    )
