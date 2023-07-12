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
        weights=[0.7, 0.3],
        benchmark_prices=benchmark,
        benchmark_weights=[0.6, 0.4],
    )

    return portfolio


def test_drawdowns(portfolio):
    dds = portfolio.drawdowns()

    assert np.all(
        np.abs(
            (
                dds.values
                - np.array(
                    [
                        [0.0],
                        [0.0],
                        [-0.00627722],
                        [-0.01553499],
                        [-0.02498078],
                        [-0.02290498],
                        [-0.02668493],
                        [-0.02769109],
                        [-0.02872094],
                        [-0.02055915],
                        [-0.01245403],
                        [-0.00469855],
                        [-0.02603059],
                        [-0.02642086],
                        [-0.03097002],
                        [-0.03638353],
                        [-0.04056918],
                        [-0.04182572],
                        [-0.04561755],
                        [-0.05809403],
                    ]
                )
            )
        )
        < 0.01
    )


def test_mdd(portfolio):
    mdd = portfolio.maximum_drawdown()
    assert np.abs(mdd - 0.058093749465727096) < 0.01


def test_add(portfolio):
    add = portfolio.average_drawdown()
    assert np.abs(add - 0.02482081719019157) < 0.01


def test_sorted_drawdowns(portfolio):
    sdds = portfolio.sorted_drawdowns()

    assert np.all(
        np.abs(
            (
                sdds
                - np.array(
                    [
                        [0.0],
                        [0.0],
                        [-0.00469855],
                        [-0.00627722],
                        [-0.01245403],
                        [-0.01553499],
                        [-0.02055915],
                        [-0.02290498],
                        [-0.02498078],
                        [-0.02603059],
                        [-0.02642086],
                        [-0.02668493],
                        [-0.02769109],
                        [-0.02872094],
                        [-0.03097002],
                        [-0.03638353],
                        [-0.04056918],
                        [-0.04182572],
                        [-0.04561755],
                        [-0.05809403],
                    ]
                )
            )
        )
        < 0.01
    )
