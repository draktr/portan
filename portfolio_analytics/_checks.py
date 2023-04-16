import numpy as np
import pandas as pd
import warnings


def _check_rate_arguments(
    annual_mar=None, annual_rfr=None, annual=None, compounding=None
):
    if not isinstance(annual_mar, (float, type(None))):
        raise ValueError("`annual_mar` should be of type `float`")
    if not isinstance(annual_rfr, (float, type(None))):
        raise ValueError("`annual_rfr` should be of type `float`")
    if not isinstance(annual, (bool, type(None))):
        raise ValueError("`annual` should be of type `bool`")
    if not isinstance(compounding, (bool, type(None))):
        raise ValueError("`compounding` should be of type `bool`")

    if annual is not None and compounding is not None:
        if not annual and compounding:
            raise ValueError(
                "Mean returns cannot be compounded if `annual` is `False`."
            )


def _check_plot_arguments(show, save):
    if not isinstance(show, bool):
        raise ValueError("`show` should be of type `bool`")
    if not isinstance(save, bool):
        raise ValueError("`save` should be of type `bool`")


def _check_periods(periods, state):
    if not isinstance(periods, int):
        raise ValueError("`periods` should be of type `int`")

    if periods >= state.shape[0]:
        periods = state.shape[0]
        warnings.warn(
            f"`periods` is larger than the number of datapoints. `periods` taken as {periods}."
        )

    return periods


def _check_posints(**kwargs):
    for name, value in kwargs.items():
        if not isinstance(value, int):
            raise ValueError(f"`{name}` should be of type `int`")
        if value < 1:
            raise ValueError(f"`{name}` should be positive")


def _check_percentage(percentage):
    if not isinstance(percentage, bool):
        raise ValueError("`percentage`  should be of type `bool`.")


def _check_multiple_returns(returns):
    if not isinstance(returns, (np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError(
            "`returns` should be of type `np.ndarray`, `pd.DataFrame` or `pd.Series`"
        )
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)
    if returns.shape[1] == 1:
        warnings.warn("Returns are provided for only one portfolio.", UserWarning)
    if np.any(np.isnan(returns)):
        warnings.warn(
            "`returns` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these.",
            UserWarning,
        )

    return returns


def _check_init(
    prices,
    weights,
    benchmark_prices,
    benchmark_weights,
    name,
    benchmark_name,
    initial_aum,
    frequency,
):
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("`prices` should be of type `pd.DataFrame`")
    if np.any(np.isnan(prices)):
        raise ValueError(
            "`prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )
    if np.any(np.isinf(prices)):
        raise ValueError(
            "`prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
        )
    if not isinstance(weights, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError(
            "`weights` should be of type `list`, `np.ndarray`, `pd.DataFrame` or `pd.Series`"
        )
    if not isinstance(benchmark_prices, pd.DataFrame):
        raise ValueError("`benchmark_prices` should be of type `pd.DataFrame`")
    if np.any(np.isnan(benchmark_prices)):
        raise ValueError(
            "`benchmark_prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )
    if np.any(np.isinf(benchmark_prices)):
        raise ValueError(
            "`benchmark_prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
        )
    if not isinstance(benchmark_weights, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError(
            "`benchmark_weights` should be of type `list`, `np.ndarray`, `pd.DataFrame` or `pd.Series`"
        )
    if not isinstance(name, str):
        raise ValueError("`name` should be of type `str`")
    if not isinstance(benchmark_name, str):
        raise ValueError("`benchmark_name` should be of type `str`")
    if not isinstance(initial_aum, (int, float)):
        raise ValueError("`initial_aum` should be of type `str`")
    if initial_aum <= 0:
        raise ValueError("`initial_aum` should be positive")
    if not isinstance(frequency, (int, float)):
        raise ValueError("`frequency` should be of type `int` or `float`")
    if prices.shape[0] != benchmark_prices.shape[0]:
        raise ValueError(
            "`prices` should have the same number of datapoints as `benchmark_prices`"
        )


def _check_mar_bounds(annual_mar_lower_bound, annual_mar_upper_bound):
    if not isinstance(annual_mar_lower_bound, (int, float)):
        raise ValueError("`annual_mar_lower_bound` should be of type `int` or `float")
    if not isinstance(annual_mar_upper_bound, (int, float)):
        raise ValueError("`annual_mar_upper_bound` should be of type `int` or `float")
    if annual_mar_lower_bound >= annual_mar_upper_bound:
        raise ValueError(
            "`annual_mar_lower_bound` should be lower than `annual_mar_upper_bound`"
        )


def _check_array_lengths(array_one, array_two):
    if len(array_one) != len(array_two):
        warnings.warn(
            "Two arrays to be concatenated are not of the same length. Note that this will result in som `NaN` values",
            UserWarning,
        )


def _check_sharpe(adjusted, probabilistic):
    if adjusted and probabilistic:
        raise ValueError()
