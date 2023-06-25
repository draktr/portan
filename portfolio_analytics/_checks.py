import numpy as np
import pandas as pd
import warnings
import numbers


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
    if not isinstance(prices, (list, np.ndarray, pd.DataFrame, pd.Series)):
        raise ValueError(
            "`prices` should be of type `list`, `np.ndarray`, `pd.DataFrame` or `pd.Series`"
        )
    if isinstance(prices, (list, np.ndarray)):
        prices = pd.DataFrame(prices)
    if np.any(np.isnan(prices)):
        raise ValueError(
            "`prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )
    if np.any(np.isinf(prices)):
        raise ValueError(
            "`prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
        )
    if prices.shape[1] == 1:
        warnings.warn(
            "Your portfolio contains only one asset, check if this is intended"
        )

    if isinstance(weights, (pd.DataFrame, pd.Series)):
        weights = weights.to_numpy()
    elif isinstance(weights, list):
        weights = np.array(weights)
    elif isinstance(weights, np.ndarray):
        pass
    else:
        raise ValueError(
            "`weights` should be of type `list`, `np.ndarray`, `pd.DataFrame` or `pd.Series`"
        )

    if not isinstance(
        benchmark_prices, (list, np.ndarray, pd.DataFrame, pd.Series, type(None))
    ):
        raise ValueError(
            "`benchmark_prices` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )
    if isinstance(benchmark_prices, (list, np.ndarray)):
        benchmark_prices = pd.DataFrame(benchmark_prices)
    if isinstance(benchmark_prices, type(None)):
        warnings.warn(
            "Benchmark is not set. To calculate analytics that require benchmark (e.g. `capm()`) you will need to set it through the method. Once set, benchmark would be saved and used for other analytics without the need to set it again."
        )
    if np.any(np.isnan(benchmark_prices)):
        raise ValueError(
            "`benchmark_prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )
    if np.any(np.isinf(benchmark_prices)):
        raise ValueError(
            "`benchmark_prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
        )

    if isinstance(benchmark_weights, (pd.DataFrame, pd.Series)):
        benchmark_weights = benchmark_weights.to_numpy()
    elif isinstance(benchmark_weights, list):
        benchmark_weights = np.array(benchmark_weights)
    elif isinstance(benchmark_weights, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "`benchmark_weights` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )

    if not isinstance(name, str):
        raise ValueError("`name` should be of type `str`")

    if not isinstance(benchmark_name, (str, type(None))):
        raise ValueError("`benchmark_name` should be of type `str` or `NoneType`")

    if not isinstance(initial_aum, numbers.Real):
        raise ValueError("`initial_aum` should be a positive real number")
    if initial_aum <= 0:
        raise ValueError("`initial_aum` should be positive")

    if not isinstance(frequency, numbers.Real):
        raise ValueError("`frequency` should be a positive real number")
    if frequency <= 0:
        raise ValueError("`frequency` should be positive")

    if prices.shape[1] != weights.shape[0]:
        raise ValueError(
            "Number of assets prices doesn't match the number of weights provided"
        )

    elif benchmark_prices is None and benchmark_weights is not None:
        raise ValueError(
            "`benchmark_prices` is not provided, while `benchmark_weights` is provided. Please provide either both arguments (to access all methods) or none of the two arguments (to access only methods that do not require benchmark). Note that benchmark can also be set by providing `benchmark_prices` and `benchmark_weights` to any relevant method."
        )
    if benchmark_prices is not None and benchmark_weights is None:
        raise ValueError(
            "`benchmark_weights` is not provided, while `benchmark_prices` is provided. Please provide either both arguments (to access all methods) or none of the two arguments (to access only methods that do not require benchmark). Note that benchmark can also be set by providing `benchmark_prices` and `benchmark_weights` to any relevant method."
        )

    if benchmark_prices is not None:
        if prices.shape[0] != benchmark_prices.shape[0]:
            raise ValueError(
                "`prices` should have the same number of datapoints as `benchmark_prices`"
            )
        if benchmark_prices.shape[1] != benchmark_weights.shape[0]:
            raise ValueError(
                "Number of benchmark prices doesn't match the number of benchmark weights provided"
            )

    return prices, weights, benchmark_prices, benchmark_weights


def _check_benchmark(
    slf_benchmark_prices, benchmark_prices, benchmark_weights, benchmark_name
):
    if benchmark_prices is not None and benchmark_weights is not None:
        set_benchmark = True
        if slf_benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
    elif benchmark_prices is None and benchmark_weights is not None:
        warnings.warn(
            "`benchmark_prices` not provided. Both `benchmark_prices` and `benchmark_weights` need to be provided for benchmark to be set/reset."
        )
        set_benchmark = False
    elif benchmark_prices is not None and benchmark_weights is None:
        warnings.warn(
            "`benchmark_weights` not provided. Both `benchmark_prices` and `benchmark_weights` need to be provided for benchmark to be set/reset."
        )
        set_benchmark = False
    else:
        set_benchmark = False

    if benchmark_prices.shape[1] != benchmark_weights.shape[0]:
        raise ValueError(
            "Number of benchmark prices doesn't match the number of benchmark weights provided"
        )

    return set_benchmark


####


def _check_rate_arguments(
    annual_mar=None, annual_rfr=None, annual=None, compounding=None
):
    if not isinstance(annual_mar, (numbers.Real, type(None))):
        raise ValueError("`annual_mar` should be a real number")
    if not isinstance(annual_rfr, (numbers.Real, type(None))):
        raise ValueError("`annual_rfr` should be a real number")
    if not isinstance(annual, (bool, type(None))):
        raise ValueError("`annual` should be of type `bool`")
    if not isinstance(compounding, (bool, type(None))):
        raise ValueError("`compounding` should be of type `bool`")


def _check_plot_arguments(show, save):
    if not isinstance(show, bool):
        raise ValueError("`show` should be of type `bool`")
    if not isinstance(save, bool):
        raise ValueError("`save` should be of type `bool`")


def _check_omega_multiple_returns(returns):
    if not isinstance(returns, (np.ndarray, pd.DataFrame)):
        raise ValueError("`returns` should be of type `np.ndarray` or `pd.DataFrame`")
    if isinstance(returns, np.ndarray):
        returns = pd.DataFrame(returns)
    if returns.shape[1] == 1:
        warnings.warn(
            "Returns are provided for only one portfolio. Only one curve will be plotted.",
            UserWarning,
        )
    if np.any(np.isnan(returns)):
        raise ValueError(
            "`returns` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )

    return returns


def _check_array_lengths(array_one, array_two):
    if len(array_one) != len(array_two):
        warnings.warn(
            "Two arrays to be concatenated are not of the same length. Note that this will result in some `NaN` values",
            UserWarning,
        )


def _check_periods(periods, state):
    if not isinstance(periods, int):
        raise ValueError("`periods` should be of type `int`")

    if periods >= state.shape[0]:
        periods = state.shape[0]
        warnings.warn(
            f"`periods` is larger than the number of datapoints. `periods` taken as {periods}."
        )

    return periods


def _check_mar_bounds(annual_mar_lower_bound, annual_mar_upper_bound):
    if not isinstance(annual_mar_lower_bound, (numbers.Real)):
        raise ValueError("`annual_mar_lower_bound` should be a real number")
    if not isinstance(annual_mar_upper_bound, (numbers.Real)):
        raise ValueError("`annual_mar_upper_bound` should be a real number")
    if annual_mar_lower_bound >= annual_mar_upper_bound:
        raise ValueError(
            "`annual_mar_lower_bound` should be lower than `annual_mar_upper_bound`"
        )


def _check_sharpe(adjusted, probabilistic):
    if adjusted and probabilistic:
        raise ValueError(
            "`adjusted` and `probabilistic` arguments cannot both be `True`"
        )


def _check_booleans(**kwargs):
    for name, value in kwargs.items():
        if not isinstance(value, bool):
            raise ValueError(f"`{name}` should be a boolean")


def _check_posints(**kwargs):
    for name, value in kwargs.items():
        if not isinstance(value, int):
            raise ValueError(f"`{name}` should be of type `int`")
        if value < 1:
            raise ValueError(f"`{name}` should be positive")


def _check_nonnegints(**kwargs):
    for name, value in kwargs.items():
        if not isinstance(value, int):
            raise ValueError(f"`{name}` should be of type `int`")
        if value < 0:
            raise ValueError(f"`{name}` should be positive")
