import numpy as np
import pandas as pd
import warnings
import numbers
from portfolio_analytics.get_data import GetData


def _check_init(
    tickers,
    prices,
    weights,
    benchmark_tickers,
    benchmark_prices,
    benchmark_weights,
    name,
    benchmark_name,
    initial_aum,
    frequency,
    start,
    end,
    interval,
    **kwargs,
):
    #####
    # datatype checks
    #####

    # tickers
    if isinstance(tickers, (pd.DataFrame, pd.Series)):
        tickers = tickers.to_numpy()
    elif isinstance(tickers, list):
        tickers = np.array(tickers)
    elif isinstance(tickers, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "`tickers` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )

    # prices
    if not isinstance(prices, (list, np.ndarray, pd.DataFrame, pd.Series, type(None))):
        raise ValueError(
            "`prices` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )
    if isinstance(prices, (list, np.ndarray)):
        prices = pd.DataFrame(prices)

    # weights
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

    # benchmark tickers
    if isinstance(benchmark_tickers, (pd.DataFrame, pd.Series)):
        benchmark_tickers = benchmark_tickers.to_numpy()
    elif isinstance(benchmark_tickers, list):
        benchmark_tickers = np.array(benchmark_tickers)
    elif isinstance(benchmark_tickers, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "`benchmark_tickers` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )

    # benchmark prices
    if not isinstance(
        benchmark_prices, (list, np.ndarray, pd.DataFrame, pd.Series, type(None))
    ):
        raise ValueError(
            "`benchmark_prices` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )
    if isinstance(benchmark_prices, (list, np.ndarray)):
        benchmark_prices = pd.DataFrame(benchmark_prices)
    elif isinstance(benchmark_prices, type(None)):
        warnings.warn(
            "Benchmark is not set. To calculate analytics that require benchmark (e.g. `capm()`) you will need to set it through the method. Once set, benchmark would be saved and used for other analytics without the need to set it again."
        )

    # benchmark weights
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

    # name
    if not isinstance(name, str):
        raise ValueError("`name` should be of type `str`")

    # benchmark name
    if not isinstance(benchmark_name, (str, type(None))):
        raise ValueError("`benchmark_name` should be of type `str` or `NoneType`")

    # initial aum
    if not isinstance(initial_aum, numbers.Real):
        raise ValueError("`initial_aum` should be a positive real number")
    if initial_aum <= 0:
        raise ValueError("`initial_aum` should be positive")

    # frequency
    if not isinstance(frequency, numbers.Real):
        raise ValueError("`frequency` should be a positive real number")
    if frequency <= 0:
        raise ValueError("`frequency` should be positive")

    #####
    # important checks
    #####

    if weights is None:
        raise ValueError("Portfolio weights are not provided.")

    if tickers is None and prices is None:
        raise ValueError("Provide either `tickers` or `prices` argument.")
    elif tickers is not None and prices is not None:
        raise ValueError(
            "Both `tickers` and `prices` arguments were provided. Provide only one to avoid clashes. If only `prices` is provided, tickers will be inferred from column names"
        )
    elif tickers is not None and prices is None:
        prices = GetData(tickers, start, end, interval, **kwargs).close

    if prices.shape[1] != weights.shape[0]:
        raise ValueError(
            "Number of assets prices doesn't match the number of weights provided"
        )

    #####
    # content checks
    #####

    if prices.shape[1] == 1:
        warnings.warn(
            "Your portfolio contains only one asset, check if this is intended"
        )

    if np.any(np.isnan(prices.fillna(method="bfill"))):
        raise ValueError(
            "`prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )
    if np.any(np.isnan(prices.fillna(method="ffill"))):
        prices = prices.fillna(method="ffill").dropna()
        warnings.warn(
            "Leading rows containing `NaN` values were removed. This is likely due to different assets being listed for different periods of time. Alternatively, you can interpolate those values using `fill_nan()` function or change `start` and `end` arguments if providing `tickers`, or provide a specific `prices` dataframe"
        )

    if np.any(np.isinf(prices)):
        raise ValueError(
            "`prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
        )

    #####
    # benchmark checks
    #####

    if (
        benchmark_tickers is None
        and benchmark_prices is None
        and benchmark_weights is None
    ):
        warnings.warn(
            "Benchmark details weren't provided. To use methods that require benchmark, pass the benchmark detail to the relevant method. Once set, benchmark doesn't need to be set again (but can be reset)."
        )
        return prices, weights, benchmark_prices, benchmark_weights
    elif (
        benchmark_tickers is None
        and benchmark_prices is None
        and benchmark_weights is not None
    ):
        raise ValueError(
            "Benchmark tickers and prices aren't provided, but benchmark weights are. To use the methods that require benchmark, provide benchmark tickers or benchmark prices."
        )
    elif benchmark_tickers is not None or benchmark_prices is not None:
        if benchmark_tickers is not None and benchmark_prices is not None:
            raise ValueError(
                "Both `benchmark_tickers` and `benchmark_prices` arguments were provided. Provide only one to avoid clashes. If only `benchmark_prices` is provided, tickers will be inferred from column names"
            )
        if benchmark_tickers is not None and benchmark_prices is None:
            benchmark_prices = GetData(
                benchmark_tickers, start, end, interval, **kwargs
            ).close

        if benchmark_weights is None:
            if benchmark_tickers.shape[0] == 1 or benchmark_prices.shape[1] == 1:
                benchmark_weights = np.array([1])
            else:
                raise ValueError(
                    "Benchmark weights are not provided, but benchmark tickers or prices are. To use the methods that require benchmark, provide benchmark weights."
                )

        if benchmark_prices.shape[1] != benchmark_weights.shape[0]:
            raise ValueError(
                "Number of benchmark asset prices doesn't match the number of benchmark weights provided"
            )
        #####
        # benchmark content
        #####
        if np.any(np.isnan(benchmark_prices.fillna(method="bfill"))):
            raise ValueError(
                "`benchmark_prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
            )
        if np.any(np.isnan(benchmark_prices.fillna(method="ffill"))):
            benchmark_prices = benchmark_prices.fillna(method="ffill").dropna()
            warnings.warn(
                "Leading rows containing `NaN` values were removed. This is likely due to different assets being listed for different periods of time. Alternatively, you can interpolate those values using `fill_nan()` function or change `start` and `end` arguments if providing `tickers`, or provide a specific `benchmark_prices` dataframe"
            )

        if np.any(np.isinf(benchmark_prices)):
            raise ValueError(
                "`benchmark_prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
            )
        #####
        # matching checks
        #####
        if benchmark_prices is not None:
            if prices.shape[0] != benchmark_prices.shape[0]:
                benchmark_prices = benchmark_prices.iloc[-prices.shape[0] :]
                prices = prices.iloc[-benchmark_prices.shape[0] :]
                warnings.warn(
                    "`prices` and `benchmark_prices` row lengths didn't match so the longer dataframe was shoretened by removing the excess leading rows."
                )
        return prices, weights, benchmark_prices, benchmark_weights


def _check_benchmark(
    benchmark_tickers,
    benchmark_prices,
    benchmark_weights,
    benchmark_name,
    prices,
    start,
    end,
    interval,
):
    #####
    # checking datatypes
    #####

    # benchmark tickers
    if isinstance(benchmark_tickers, (pd.DataFrame, pd.Series)):
        benchmark_tickers = benchmark_tickers.to_numpy()
    elif isinstance(benchmark_tickers, list):
        benchmark_tickers = np.array(benchmark_tickers)
    elif isinstance(benchmark_tickers, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "`benchmark_tickers` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )

    # benchmark prices
    if not isinstance(
        benchmark_prices, (list, np.ndarray, pd.DataFrame, pd.Series, type(None))
    ):
        raise ValueError(
            "`benchmark_prices` should be of type `list`, `np.ndarray`, `pd.DataFrame`, `pd.Series` or `NoneType`"
        )
    if isinstance(benchmark_prices, (list, np.ndarray)):
        benchmark_prices = pd.DataFrame(benchmark_prices)
    elif isinstance(benchmark_prices, type(None)):
        warnings.warn(
            "Benchmark is not set. To calculate analytics that require benchmark (e.g. `capm()`) you will need to set it through the method. Once set, benchmark would be saved and used for other analytics without the need to set it again."
        )

    # benchmark weights
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

    # benchmark name
    if not isinstance(benchmark_name, (str, type(None))):
        raise ValueError("`benchmark_name` should be of type `str` or `NoneType`")

    ###
    if (
        benchmark_tickers is None
        and benchmark_prices is None
        and benchmark_weights is not None
    ):
        raise ValueError(
            "Benchmark tickers and prices aren't provided, but benchmark weights are. To use the methods that require benchmark, provide benchmark tickers or benchmark prices."
        )
    elif benchmark_tickers is not None or benchmark_prices is not None:
        if benchmark_tickers is not None and benchmark_prices is not None:
            raise ValueError(
                "Both `benchmark_tickers` and `benchmark_prices` arguments were provided. Provide only one to avoid clashes. If only `benchmark_prices` is provided, tickers will be inferred from column names"
            )
        if benchmark_tickers is not None and benchmark_prices is None:
            benchmark_prices = GetData(benchmark_tickers, start, end, interval).close
        if benchmark_weights is None:
            if benchmark_tickers.shape[0] == 1 or benchmark_prices.shape[1] == 1:
                benchmark_weights = np.array([1])
            else:
                raise ValueError(
                    "Benchmark weights are not provided, but benchmark tickers or prices are. To use the methods that require benchmark, provide benchmark weights."
                )
        if benchmark_prices.shape[1] != benchmark_weights.shape[0]:
            raise ValueError(
                "Number of benchmark asset prices doesn't match the number of benchmark weights provided"
            )
        #####
        # benchmark content
        #####
        if np.any(np.isnan(benchmark_prices.fillna(method="bfill"))):
            raise ValueError(
                "`benchmark_prices` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
            )
        if np.any(np.isnan(benchmark_prices.fillna(method="ffill"))):
            benchmark_prices = benchmark_prices.fillna(method="ffill").dropna()
            warnings.warn(
                "Leading rows containing `NaN` values were removed. This is likely due to different assets being listed for different periods of time. Alternatively, you can interpolate those values using `fill_nan()` function or change `start` and `end` arguments if providing `tickers`, or provide a specific `benchmark_prices` dataframe"
            )

        if np.any(np.isinf(benchmark_prices)):
            raise ValueError(
                "`benchmark_prices` contains `inf` values. Use `fill_inf()` from `utilities` module to interpolate these."
            )
        #####
        # matching checks
        #####
        if benchmark_prices is not None:
            if prices.shape[0] != benchmark_prices.shape[0]:
                benchmark_prices = benchmark_prices.iloc[-prices.shape[0] :]
                prices = prices.iloc[-benchmark_prices.shape[0] :]
                warnings.warn(
                    "`prices` and `benchmark_prices` row lengths didn't match so the longer dataframe was shoretened by removing the excess leading rows."
                )

        return benchmark_prices, benchmark_weights, prices


def _whether_to_set(
    slf_benchmark_prices,
    benchmark_tickers,
    benchmark_prices,
    benchmark_weights,
    benchmark_name,
    start,
    end,
    interval,
):
    if (
        benchmark_tickers is None
        and benchmark_prices is None
        and slf_benchmark_prices is None
    ):
        raise ValueError(
            "This method cannot be used as benchmark is not set. Set the benchmark by passing the relevant arguments to the method. Once set, benchmark doesn't need to be set again (but can be reset/changed)."
        )
    if (
        benchmark_tickers is not None
        or benchmark_prices is not None
        and benchmark_weights is not None
    ):
        if slf_benchmark_prices is not None:
            warnings.warn(
                "By providing `benchmark_prices` and `benchmark_weights` you are resetting the benchmark"
            )
        if (
            benchmark_tickers.shape[0] != benchmark_weights.shape[0]
            or benchmark_prices.shape[1] != benchmark_weights.shape[0]
        ):
            raise ValueError(
                "Number of benchmark assets doesn't match the number of benchmark weights provided"
            )
        set_benchmark = True
    elif (
        benchmark_tickers is None
        and benchmark_prices is None
        and benchmark_weights is not None
    ):
        raise ValueError(
            "Benchmark assets aren't provided, but benchmark weights are. To use the methods that require benchmark, provide benchmark tickers or benchmark prices."
        )
    elif (
        benchmark_tickers is not None
        or benchmark_prices is not None
        and benchmark_weights is None
    ):
        raise ValueError(
            "Benchmark weights are not provided, but benchmark assets are. To use the methods that require benchmark, provide benchmark weights."
        )

    else:
        set_benchmark = False

    return set_benchmark


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
    if not isinstance(returns, (list, np.ndarray, pd.DataFrame)):
        raise ValueError(
            "`returns` should be of type `list`, `np.ndarray` or `pd.DataFrame`"
        )
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.DataFrame(returns)
    if returns.shape[1] == 1:
        warnings.warn(
            "Returns are provided for only one portfolio. Only one curve will be plotted."
        )
    if np.any(np.isnan(returns)):
        raise ValueError(
            "`returns` contains `NaN` values. Use `fill_nan()` from `utilities` module to interpolate these."
        )

    return returns


def _check_array_lengths(array_one, array_two):
    if len(array_one) != len(array_two):
        warnings.warn(
            "Two arrays to be concatenated are not of the same length. Note that this will result in some `NaN` values"
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
