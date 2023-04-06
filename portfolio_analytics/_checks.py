import numpy as np


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
