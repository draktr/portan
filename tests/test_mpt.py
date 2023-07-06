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


def test_excess_mar(portfolio):
    m = portfolio.excess_mar()

    assert np.abs(m - -0.5690610086327291) < 0.01


def test_sharpe(portfolio):
    m = portfolio.sharpe()

    assert np.abs(m - -4.841868878721975) < 0.01


def test_capm_return(portfolio):
    m = portfolio.capm_return()

    assert np.abs(m - 0.12386894643187883) < 0.01


def test_capm(portfolio):
    m = portfolio.capm()

    assert (
        np.abs(m[0] - -0.0034775343143376193) < 0.01
        and np.abs(m[1] - 0.3089406560051947) < 0.01
        and np.all(
            np.abs(
                m[2].values
                - np.array(
                    [
                        [0.00541534],
                        [-0.00321467],
                        [-0.00643383],
                        [-0.00655332],
                        [0.00466349],
                        [-0.00157202],
                        [0.00234979],
                        [0.00220123],
                        [0.01120238],
                        [0.01075782],
                        [0.0116929],
                        [-0.01729616],
                        [0.00336265],
                        [-0.00153086],
                        [-0.00402883],
                        [-0.0012814],
                        [0.00154481],
                        [-0.00082123],
                        [-0.01045808],
                    ]
                )
            )
            < 0.01
        )
        and np.abs(m[3] - 0.007103294978593522) < 0.01
    )


def test_ewm(portfolio):
    m = portfolio.ewm(alpha=0.3)

    assert np.abs(m - -0.7975404206500287) < 0.01


def test_distribution_test(portfolio):
    m = portfolio.distribution_test()

    assert np.abs(m[0][0] - 2.21501363) < 0.01 and np.abs(m[1][0] - 0.33038164) < 0.01


def test_net_return(portfolio):
    m = portfolio.net_return()

    assert np.abs(m - -568.5290375899458) < 0.01


def test_excess_benchmark(portfolio):
    m = portfolio.excess_benchmark()

    assert np.abs(m - -0.8728706795221505) < 0.01
