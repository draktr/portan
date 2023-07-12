import pytest
import numpy as np
import pandas as pd
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


def test_assets_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.assets_returns["GOOG"].values
            - np.array(
                [
                    0.00431311,
                    -0.01387143,
                    -0.01364159,
                    -0.04239875,
                    0.00109243,
                    0.0045255,
                    0.00587892,
                    -0.00738513,
                    0.00574408,
                    0.00688857,
                    0.01052277,
                    -0.08377501,
                    -0.00080208,
                    -0.00783914,
                    -0.01969257,
                    -0.00244085,
                    0.02091187,
                    -0.00394842,
                    0.0041891,
                ]
            )
        )
        < 0.01
    ) & np.all(
        np.abs(
            portfolio.assets_returns["XOM"].values
            - np.array(
                [
                    0.00023236,
                    -0.00302249,
                    -0.0074626,
                    0.00446454,
                    0.00257286,
                    -0.00746629,
                    -0.0039961,
                    0.00165202,
                    0.00954294,
                    0.00886919,
                    0.00670894,
                    0.00528557,
                    -0.00022848,
                    -0.00331546,
                    0.00045878,
                    -0.00515947,
                    -0.01083333,
                    -0.00396083,
                    -0.02047052,
                ]
            )
        )
        < 0.01
    )


def test_assets_names(portfolio):
    assert np.all(
        portfolio.assets_names == ["Exxon Mobil Corporation", "Alphabet Inc."]
    )


def test_allocation_funds(portfolio):
    assert np.all(portfolio.allocation_funds.values == np.array([7000.0, 3000.0]))


def test_allocation_assets(portfolio):
    assert np.all(
        np.abs(
            portfolio.allocation_assets.values - np.array([129.561992, 181.0159012])
            < 0.01
        )
    )


def test_state(portfolio):
    assert (
        np.all(
            np.abs(
                portfolio.state["GOOG"].values
                - np.array(
                    [
                        3000.0,
                        3012.9393252,
                        2971.14554994,
                        2930.61439209,
                        2806.35999638,
                        2809.42573638,
                        2822.13977915,
                        2838.73092393,
                        2817.76653789,
                        2833.95200166,
                        2853.47388962,
                        2883.50032621,
                        2641.93506733,
                        2639.81603143,
                        2619.12215694,
                        2567.54490946,
                        2561.27791474,
                        2614.83902865,
                        2604.51453334,
                        2615.42510824,
                    ]
                )
            )
            < 0.01
        )
        & np.all(
            np.abs(
                portfolio.state["XOM"].values
                - np.array(
                    [
                        7000.0,
                        7001.62407191,
                        6980.46369011,
                        6928.37032341,
                        6959.30231983,
                        6977.20664998,
                        6925.11377753,
                        6897.44030395,
                        6908.83450776,
                        6974.76757668,
                        7036.62563865,
                        7083.83393939,
                        7121.27705063,
                        7119.6490248,
                        7096.04561579,
                        7099.29969048,
                        7062.67207487,
                        6986.16029778,
                        6958.48484724,
                        6816.0434609,
                    ]
                )
            )
            < 0.01,
        )
        & np.all(
            np.abs(
                portfolio.state[portfolio.name].values
                == np.array(
                    [
                        10000.0,
                        10014.56586888,
                        9951.6097274,
                        9858.98617294,
                        9765.66378456,
                        9786.63484952,
                        9747.25501296,
                        9736.1726744,
                        9726.60299043,
                        9808.7190752,
                        9890.10151816,
                        9967.33627216,
                        9763.21314925,
                        9759.46707543,
                        9715.16830087,
                        9666.84661196,
                        9623.95100022,
                        9600.99981578,
                        9563.00430832,
                        9431.4709754,
                    ]
                )
            )
        )
        < 0.01
    )


def test_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.returns
            - np.array(
                [
                    [0.00145634],
                    [-0.00627698],
                    [-0.00931639],
                    [-0.00959445],
                    [0.00212863],
                    [-0.00386866],
                    [-0.0010336],
                    [-0.00105918],
                    [0.00840353],
                    [0.00827476],
                    [0.00785309],
                    [-0.0214325],
                    [-0.00040065],
                    [-0.00467241],
                    [-0.00558677],
                    [-0.00434378],
                    [-0.00130972],
                    [-0.00395756],
                    [-0.01307239],
                ]
            )
        )
        < 0.01
    )


def test_cumulative_returns(portfolio):
    assert np.all(
        np.abs(
            portfolio.cumulative_returns
            - np.array(
                [
                    [1.00145634],
                    [0.99517022],
                    [0.98589882],
                    [0.97643967],
                    [0.97851815],
                    [0.9747326],
                    [0.97372512],
                    [0.97269377],
                    [0.98086784],
                    [0.98898428],
                    [0.99675086],
                    [0.97538799],
                    [0.9749972],
                    [0.97044161],
                    [0.96501998],
                    [0.96082814],
                    [0.95956972],
                    [0.95577217],
                    [0.94327794],
                ]
            )
        )
        < 0.01
    )


def test_mean(portfolio):
    assert np.abs(portfolio.mean - -0.0030425625869149532) < 0.01


def test_arithmetic_mean(portfolio):
    assert np.abs(portfolio.arithmetic_mean - -0.7667257719025682) < 0.01


def test_geometric_mean(portfolio):
    assert np.abs(portfolio.geometric_mean - -0.53906246664295) < 0.01


def test_volatility(portfolio):
    assert np.abs(portfolio.volatility - 0.0074035378303970705) < 0.01


def test_annual_volatility(portfolio):
    assert np.abs(portfolio.annual_volatility - 0.11752964613162667) < 0.01


def test_skewness(portfolio):
    assert np.abs(portfolio.skewness - -0.4645961569495024) < 0.01


def test_kurtosis(portfolio):
    assert np.abs(portfolio.kurtosis - 0.45332311551375915) < 0.01


def test_min_aum(portfolio):
    assert np.abs(portfolio.min_aum - 9431.46856914042) < 0.01


def test_max_aum(portfolio):
    assert np.abs(portfolio.max_aum - 10014.56339710365) < 0.01


def test_mean_aum(portfolio):
    assert np.abs(portfolio.mean_aum - 9768.886903514209) < 0.01


def test_final_aum(portfolio):
    assert np.abs(portfolio.final_aum - 9431.46856914042) < 0.01
