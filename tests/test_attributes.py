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
                    0.00023201,
                    -0.00302221,
                    -0.00746274,
                    0.00446454,
                    0.00257272,
                    -0.00746615,
                    -0.0039961,
                    0.00165195,
                    0.0095433,
                    0.00886883,
                    0.00670894,
                    0.00528571,
                    -0.00022861,
                    -0.00331525,
                    0.00045858,
                    -0.00515933,
                    -0.01083326,
                    -0.00396147,
                    -0.02047017,
                ]
            )
        )
        < 0.01
    )


def test_assets_names(portfolio):
    assert np.all(
        portfolio.assets_names == np.array(["Alphabet Inc.", "Exxon Mobil Corporation"])
    )


def test_allocation_funds(portfolio):
    assert np.all(portfolio.allocation_funds.values == np.array([3000.0, 7000.0]))


def test_allocation_assets(portfolio):
    assert np.all(
        np.abs(portfolio.allocation_assets.values - np.array([181.0159012, 129.561992]))
        < 0.01
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
                        10014.5633971,
                        9951.60924005,
                        9858.9847155,
                        9765.66231621,
                        9786.63238636,
                        9747.25355667,
                        9736.17122788,
                        9726.60104565,
                        9808.71957835,
                        9890.09952827,
                        9967.3342656,
                        9763.21211796,
                        9759.46505623,
                        9715.16777273,
                        9666.84459994,
                        9623.94998961,
                        9600.99932643,
                        9562.99938058,
                        9431.46856914,
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
