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


def test_benchmark_assets_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.benchmark_assets_returns["ITOT"].values
            - np.array(
                [
                    0.00017253,
                    0.00241564,
                    -0.00172116,
                    0.00258595,
                    0.00997454,
                    0.00017024,
                    0.00323437,
                    -0.00559966,
                    0.00341279,
                    0.01207489,
                    0.00537731,
                    -0.00083593,
                    0.00033465,
                    0.00033446,
                    0.0076899,
                    -0.00530855,
                    -0.00016714,
                    -0.00300215,
                    0.0006691,
                ]
            )
        )
        < 0.01
    ) & np.all(
        np.abs(
            portfolio.benchmark_assets_returns["IEF"].values
            - np.array(
                [
                    -0.00314616,
                    0.00019093,
                    0.00392049,
                    0.00019086,
                    -0.00200003,
                    0.00581976,
                    -0.00208621,
                    0.00446729,
                    0.00094648,
                    -0.00312011,
                    -0.00597496,
                    -0.00343465,
                    -0.00220183,
                    0.00115121,
                    0.00479199,
                    0.00534187,
                    0.00303515,
                    0.00340597,
                    0.00377074,
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
    assert np.abs(portfolio.benchmark_mean - 0.0013218250018707618) < 0.01


def test_benchmark_arithmetic_mean(portfolio):
    assert np.abs(portfolio.benchmark_arithmetic_mean - 0.33309990047143195) < 0.01


def test_benchmark_geometric_mean(portfolio):
    assert np.abs(portfolio.benchmark_geometric_mean - 0.3940063396989859) < 0.01
