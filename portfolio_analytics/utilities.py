import numpy as np
import pandas as pd
from portfolio_analytics.get_data import GetData
from portfolio_analytics._checks import _check_array_lengths


def concatenate_portfolios(portfolio_one, portfolio_two):
    """
    Concatenates an array of portfolio returns to an existing array of portfolio returns.

    :param portfolio_one: Returns of first portfolio(s)
    :type portfolio_one: list, np.ndarray, pd.DataFrame, pd.Series
    :param portfolio_two: Returns of portfolio(s) to be concatenated to the right
    :type portfolio_two: list, np.ndarray, pd.DataFrame, pd.Series
    :return: DataFrame with returns of the given portfolios in their respective columns
    :rtype: pd.DataFrame
    """

    _check_array_lengths(array_one=portfolio_one, array_two=portfolio_two)

    portfolios = pd.concat(
        [pd.DataFrame(portfolio_one), pd.DataFrame(portfolio_two)], axis=1
    )

    return portfolios


def rate_conversion(given_rate, periods=252):
    """
    Changes rate given in one periodization into another periodization,
    e.g. annual rate of return into daily rate of return etc.

    :param given_rate: Rate of interest, return etc
    :type given_rate: int, float
    :param periods: How many given rate periods there is in one output rate period.
                                    Defaults to 252. Converts annual rate into daily rate given 252 trading days.
                                    `periods=365` converts annual rate into daily (calendar) rate.
                                    `periods=1/252` converts daily (trading) rate into annual rate.
                                    `periods=1/12` converts monthly rate into annual, defaults to 252
    :type periods: int, optional
    :return: Rate expressed in a specified periodization
    :rtype: float
    """

    output_rate = (given_rate + 1) ** (1 / periods) - 1

    return output_rate


def fill_nan(portfolio_returns, method="adjacent"):
    if method == "adjacent":
        portfolio_returns.interpolate(method="linear", inplace=True)
    elif method == "column":
        portfolio_returns.fillna(portfolio_returns.mean(), inplace=True)
    else:
        raise ValueError("Fill method unsupported.")

    return portfolio_returns


def multiple_portfolios(tickers, weights, **kwargs):
    data = GetData(tickers, weights, **kwargs).data["Adj Close"]
    returns = data.pct_change().drop(data.index[0])

    return returns
