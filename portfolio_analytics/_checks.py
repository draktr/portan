import numpy as np
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


def _check_period(period, portfolio_state):

    if not isinstance(period, int):
        raise ValueError()

    if period >= portfolio_state.shape[0]:
        period = portfolio_state.shape[0]
        warnings.warn("Dataset too small. Period taken as {}.".format(period))

    return period


def _check_posints(argument):
    if not isinstance(argument, int):
        raise ValueError()
    if argument < 1:
        raise ValueError()
