import pytest
import numpy as np
from portfolio_analytics.portfolio_analytics import PortfolioAnalytics
from portfolio_analytics.get_data import GetData


@pytest.fixture
def portfolio():
    data = GetData(tickers=["XOM", "GOOG"], start="2012-01-01", end="2012-02-01").data[
        "Close"
    ]
    benchmark = GetData(
        tickers=["ITOT", "IEF"], start="2012-01-01", end="2012-02-01"
    ).data["Close"]

    portfolio = PortfolioAnalytics(
        prices=data,
        weights=[0.3, 0.7],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_drawdowns(portfolio):
    dds = portfolio.drawdowns()

    assert np.all(
        np.abs(
            (
                dds
                - np.array(
                    [
                        [0.0],
                        [0.0],
                        [-0.00627712],
                        [-0.01553489],
                        [-0.02498049],
                        [-0.02290459],
                        [-0.02668469],
                        [-0.0276907],
                        [-0.02872079],
                        [-0.02055881],
                        [-0.01245436],
                        [-0.00469874],
                        [-0.02603068],
                        [-0.02642086],
                        [-0.03096973],
                        [-0.03638348],
                        [-0.04056927],
                        [-0.04182576],
                        [-0.04561765],
                        [-0.05809375],
                    ]
                )
            )
        )
        < 0.01
    )


def test_mdd(portfolio):
    mdd = portfolio.maximum_drawdown()
    assert np.abs(mdd - 0.058093749465727096) < 0.01


def test_mdd(portfolio):
    add = portfolio.average_drawdown()
    assert np.abs(add - 0.02482081719019157) < 0.01


def test_drawdowns(portfolio):
    sdds = portfolio.sorted_drawdowns()

    assert np.all(
        np.abs(
            (
                sdds
                - np.array(
                    [
                        [-0.05809375],
                        [-0.04561765],
                        [-0.04182576],
                        [-0.04056927],
                        [-0.03638348],
                        [-0.03096973],
                        [-0.02872079],
                        [-0.0276907],
                        [-0.02668469],
                        [-0.02642086],
                        [-0.02603068],
                        [-0.02498049],
                        [-0.02290459],
                        [-0.02055881],
                        [-0.01553489],
                        [-0.01245436],
                        [-0.00627712],
                        [-0.00469874],
                        [0.0],
                        [0.0],
                    ]
                )
            )
        )
        < 0.01
    )
