import numpy as np
import pandas as pd
import warnings


def _check_rate_arguments(
    annual_mar=None, annual_rfr=None, annual=None, compounding=None
):
    if not isinstance(annual_mar, (float, type(None))):
        raise ValueError()
    if not isinstance(annual_rfr, (float, type(None))):
        raise ValueError()
    if not isinstance(annual, (bool, type(None))):
        raise ValueError()
    if not isinstance(compounding, (bool, type(None))):
        raise ValueError()

    if annual is not None and compounding is not None:
        if not annual and compounding:
            raise ValueError(
                "Mean returns cannot be compounded if `annual` is `False`."
            )


def _check_plot_arguments(show, save):
    if not isinstance(show, bool):
        raise ValueError()
    if not isinstance(save, bool):
        raise ValueError()


def _check_periods(periods, state):
    if not isinstance(periods, int):
        raise ValueError()

    if periods >= state.shape[0]:
        periods = state.shape[0]
        warnings.warn("Dataset too small. `periods` taken as {}.".format(periods))

    return periods


def _check_posints(argument):
    if not isinstance(argument, int):
        raise ValueError()
    if argument < 1:
        raise ValueError()


def _check_percentage(percentage):
    if not isinstance(percentage, bool):
        raise ValueError("Percentage argument should be a `bool`.")


def _check_multiple_returns(returns):
    if not isinstance(returns, (np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError()
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)
    if returns.shape[1] == 1:
        warnings.Warn()
    if np.any(np.isnan(returns)):
        warnings.Warn()

    return returns


def _check_init(prices, weights, name, initial_aum, frequency):
    if not isinstance(prices, pd.DataFrame):
        raise ValueError()
    if np.any(np.isnan(prices)):
        raise ValueError()
    if np.any(np.isinf(prices)):
        raise ValueError()
    if not isinstance(weights, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError()
    if not isinstance(name, str):
        raise ValueError()
    if not isinstance(initial_aum, (int, float)):
        raise ValueError()
    if initial_aum <= 0:
        raise ValueError()
    if not isinstance(frequency, (int, float)):
        raise ValueError()


def _check_pt_init(
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
        raise ValueError()
    if np.any(np.isnan(prices)):
        raise ValueError()
    if np.any(np.isinf(prices)):
        raise ValueError()
    if not isinstance(weights, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError()
    if not isinstance(benchmark_prices, pd.DataFrame):
        raise ValueError()
    if not isinstance(benchmark_weights, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError()
    if not isinstance(name, str):
        raise ValueError()
    if not isinstance(benchmark_name, str):
        raise ValueError()
    if not isinstance(initial_aum, (int, float)):
        raise ValueError()
    if initial_aum <= 0:
        raise ValueError()
    if not isinstance(frequency, (int, float)):
        raise ValueError()


def _check_mar_bounds(annual_mar_lower_bound, annual_mar_upper_bound):
    if not isinstance(annual_mar_lower_bound, (int, float)):
        raise ValueError()
    if not isinstance(annual_mar_upper_bound, (int, float)):
        raise ValueError()
    if annual_mar_lower_bound >= annual_mar_upper_bound:
        raise ValueError()


def _check_array_lengths(array_one, array_two):
    if len(array_one) != len(array_two):
        warnings.warn()
