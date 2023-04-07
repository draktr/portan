import pandas as pd


def concatenate_portfolios(portfolio_one, portfolio_two):
    """
    Concatenates an array of portfolio returns to an existing array of portfolio returns.
    Accepts array-like objects such as np.ndarray, pd.DataFrame, pd.Series, list etc.

    Args:
        portfolio_one (array-like object): Returns of first portfolio(s).
        portfolio_two (array-like object): Returns of portfolio(s) to be concatenated to the right.

    Returns:
        pd.DataFrame: DataFrame with returns of the given portfolios in respective columns.
    """

    portfolios = pd.concat(
        [pd.DataFrame(portfolio_one), pd.DataFrame(portfolio_two)], axis=1
    )

    return portfolios


def periodization(given_rate, periods=1 / 252):
    """
    Changes rate given in one periodization into another periodization.
    E.g. annual rate of return into daily rate of return etc.

    Args:
        given_rate (float): Rate of interest, return etc. Specified in decimals.
        periods (float, optional): How many given rate periods there is in one output rate period.
                                    Defaults to 1/252. Converts annual rate into daily rate given 252 trading days.
                                    periods=1/365 converts annual rate into daily (calendar) rate.
                                    periods=252 converts daily (trading) rate into annual rate.
                                    periods=12 converts monthly rate into annual.

    Returns:
        float: Rate expressed in a specified period.
    """
    # TODO: should self.frequence be involved here?
    output_rate = (given_rate + 1) ** (periods) - 1

    return output_rate


def fill_nan(portfolio_returns, method="adjacent", data_object="pandas"):

    if method == "adjacent":
        portfolio_returns.interpolate(method="linear", inplace=True)
    elif method == "column":
        portfolio_returns.fillna(portfolio_returns.mean(), inplace=True)
    else:
        raise ValueError("Fill method unsupported.")


# TODO: returns of multiple portfolios at once vixen 2d array of weights and tickers
