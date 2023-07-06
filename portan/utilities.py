"""
`utilities.py` module contains functions that complement the main
features of the package. Currently implemented utilities are:
- `concatenate_portfolios()`
- `rate_conversion()`
- `fill_nan()`
- `fill_inf()`
- `multi_returns()`
"""


import numpy as np
import pandas as pd
from portan.get_data import GetData
from portan._checks import _check_array_lengths


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


def fill_nan(returns, method="adjacent"):
    """
    Fills NaN (not a number) values of a returns array

    :param returns: Returns array
    :type returns: pd.DataFrame, pd.Series
    :param method: Fill method, defaults to "adjacent"
    :type method: str, optional
    :raises ValueError: Raises error if Fill method is unsupported
    :return: Returns array with filled NaNs
    :rtype: pd.DataFrame or pd.Series
    """

    if method == "adjacent":
        returns = returns.interpolate(method="linear", inplace=True)
    elif method == "column":
        returns = returns.fillna(returns.mean(), inplace=True)
    elif method == "ffill":
        returns = returns.fillna(method="ffill", inplace=True)
    elif method == "bfill":
        returns = returns.fillna(method="bfill", inplace=True)
    else:
        raise ValueError("Fill method unsupported.")

    return returns


def fill_inf(returns, method="adjacent"):
    """
    Fills inf (-inf and +inf) values of a returns array


    :param returns: Returns array
    :type returns: pd.DataFrame, pd.Series
    :param method: Fill method, defaults to "adjacent"
    :type method: str, optional
    :return: Returns array with filled infs
    :rtype: pd.DataFrame or pd.Series
    """

    returns.replace([np.inf, -np.inf], np.nan)
    returns = fill_nan(returns=returns, method=method)

    return returns


def multi_returns(tickers, weights, **kwargs):
    """
    Calculates returns of multiple assets

    :param tickers: Array of tickers
    :type tickers: list, np.ndarray
    :param weights: Array of weights
    :type weights: list, np.ndarray
    :return: Returns array of assets
    :rtype: pd.DataFrame
    """

    data = GetData(tickers, weights, **kwargs).data["Close"]
    returns = data.pct_change().drop(data.index[0])

    return returns
