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


def test_benchmark_assets_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.benchmark_assets_returns["ITOT"].values
            - np.array(
                [
                    0.00017253,
                    0.00241531,
                    -0.00172108,
                    0.00258603,
                    0.0099743,
                    0.00017056,
                    0.0032342,
                    -0.00559982,
                    0.00341295,
                    0.01207489,
                    0.00537739,
                    -0.0008357,
                    0.00033442,
                    0.00033462,
                    0.00768966,
                    -0.00530862,
                    -0.00016667,
                    -0.00300255,
                    0.00066926,
                ]
            )
        )
        < 0.01
    ) & np.all(
        np.abs(
            portfolio.assets_returns["IEF"].values
            - np.array(
                [
                    -0.00314652,
                    0.00019227,
                    0.00391995,
                    0.00019068,
                    -0.00200039,
                    0.00582003,
                    -0.00208603,
                    0.00446729,
                    0.00094595,
                    -0.00311984,
                    -0.00597514,
                    -0.00343474,
                    -0.00220183,
                    0.00115139,
                    0.00479234,
                    0.00534133,
                    0.00303622,
                    0.00340499,
                    0.00377101,
                ]
            )
        )
        < 0.01
    )


def test_benchmark_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.benchmark_returns
            - np.array(
                [
                    [-1.81890024e-03],
                    [1.08148736e-03],
                    [1.66353863e-03],
                    [1.14882017e-03],
                    [2.78948458e-03],
                    [3.56024288e-03],
                    [4.20612225e-05],
                    [4.40445533e-04],
                    [1.93274729e-03],
                    [2.95805011e-03],
                    [-1.43412794e-03],
                    [-2.39512114e-03],
                    [-1.18733166e-03],
                    [8.24684397e-04],
                    [5.95127063e-03],
                    [1.08135092e-03],
                    [1.75506265e-03],
                    [8.41977252e-04],
                    [2.53030808e-03],
                ]
            )
        )
        < 0.01
    )


def test_benchmark_mean(portfolio):
    assert np.abs(portfolio.benchmark_mean - 0.0011455816173713998) < 0.01


def test_benchmark_arithmetic_mean(portfolio):
    assert np.abs(portfolio.benchmark_arithmetic_mean - 0.28868656757759276) < 0.01


def test_benchmark_geometric_mean(portfolio):
    assert np.abs(portfolio.benchmark_geometric_mean - 0.33380485767037205) < 0.01
