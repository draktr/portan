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


def test_excess_mar(portfolio):
    m = portfolio.excess_mar()

    assert np.abs(m - -0.5690610086327291) < 0.01


def test_sharpe(portfolio):
    m = portfolio.sharpe()

    assert np.abs(m - -4.841868878721975) < 0.01


def test_capm_return(portfolio):
    m = portfolio.capm_return()

    assert np.abs(m - 0.3883990738838218) < 0.01


def test_capm(portfolio):
    m = portfolio.capm()

    assert (
        np.abs(m[0] - -0.004345819651720528) < 0.01
        and np.abs(m[1] - 0.9845956918777815) < 0.01
        and np.all(
            np.abs(
                m[2].values
                - np.array(
                    [
                        [0.00693723],
                        [-0.00343542],
                        [-0.00549892],
                        [-0.00685407],
                        [0.00136807],
                        [-0.00191697],
                        [0.00222126],
                        [0.00483355],
                        [0.01035812],
                        [0.00671457],
                        [0.01137356],
                        [-0.01524195],
                        [0.0046128],
                        [-0.00097945],
                        [-0.00767267],
                        [0.00103254],
                        [0.00193715],
                        [0.00081919],
                        [-0.0106086],
                    ]
                )
            )
            < 0.01
        )
        and np.abs(m[3] - 0.10383814753314835) < 0.01
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

    assert np.abs(m - -0.9330672902327009) < 0.01
