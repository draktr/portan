# PortAn

`portan` is a set of portfolio analytics and reporting toolset in Python. The package is written to be comprehensive and flexible, while emphasizing ease of use.

* Free software: MIT license
* Documentation: <https://portan.readthedocs.io/en/latest/>

## Installation

Preferred method to install `portan` is through Python's package installer pip. To install `portan`, run this command in your terminal

```shell
pip install portan
```

Alternatively, you can install the package directly from GitHub:

```shell
git clone -b development https://github.com/draktr/portan.git
cd portan
python setup.py install
```

# User Guides

## 5 Minute Guide

Simplest way to initiate analysis is to pass tickers and weights arguments to `Analytics` class.

```python
from portan import Analytics

portfolio = Analytics(
    tickers=["XOM", "GOOG", "T"],
    weights=[0.3, 0.3, 0.4],
)
```

Initiation of the `Analytics` object automatically calculates some basic portfolio attributes such as return time-series, mean return (simple, geometric and arithmetic), return volatility, AUM statistics, etc. For example, to get compounded annual mean return call

```python
>>> print(f"Compounded mean return: {portfolio.geometric_mean}")
0.12978880349610256
```

However, most of the analytics can be obtained by calling the relevant method. Here are a few examples:

```python
premium = portfolio.excess_mar(annual_mar=0.05)
results = portfolio.distribution_test(test="kolomogorov-smirnov", distribution="norm")
ulcer_index = portfolio.ulcer()
var = portfolio.parametric_var()
```

This yields

```shell
>>> print(f"Premium: {premium}")
Premium: 0.07978880349610255
>>> print(f"KS Test p-value: {results[1]}")
KS Test p-value: 0.0
>>> print(f"Ulcer Index: {ulcer_index}")
Ulcer Index: 15.025392174313993
>>> print(f"Parametric Value-at-Risk: {var}")
Parametric Value-at-Risk: -0.02050514669206502
```

**IMPORTANT**: Since we didn't provide benchmark details, we cannot use any analytics that require benchmark such as portfolio $\beta$, information ratio, up capture, etc. To utilize full set of `portan.Analytics` features the user should specify benchmark details. Recommended way to do this is to provide benchmark details (benchmark weights and benchmark tickers or prices) at object instantiation:

```python
from portan import Analytics

portfolio = Analytics(
    tickers=["XOM", "GOOG", "T"],
    weights=[0.3, 0.3, 0.4],
    benchmark_tickers=["ITOT", "IEF"],
    benchmark_weights=[0.6, 0.4],
)
```

In this example, we took standard 60/40 equity-fixed income portfolio as benchmark. This allows us to calculate other analytics. For example:

```python
alpha, beta, epsilon, r_squared = portfolio.capm()
ir = portfolio.information_ratio()
treynor = portfolio.treynor()
downside_risk = portfolio.downside_risk()
```

This yields

```shell
>>> print(f"Portfolio beta: {beta}")
Portfolio beta: 1.4821305983730502
>>> print(f"Information ratio: {ir}")
Information ratio: 0.3873682857378387
>>> print(f"Treynor: {treynor}")
Treynor: 0.06732794488749151
>>> print(f"Downside risk: {downside_risk}")
Downside risk: 0.008819039607513948
```

Alternatively, benchmark can be set and reset after initiation of `Analytics` object. More on this is discussed in [(Re)setting Benchmark](#(re)setting-benchmark) section.

Features explored here are only to demonstrate basic functionality of `Analytics` class and complete list of `portan` features is listed in [Features](#features) section.
