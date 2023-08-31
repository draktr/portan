"""
`portfolios.py` contains dictionaries with tickers and weights of assets for some notable portfolio examples.

Portfolio examples taken from [PortfolioCharts](portfoliocharts.com/portfolios/):
- All Weather
- Butterfly
- 60/40 stocks and bonds
- Core Four
- Coffeehouse
- Global
- Ideal
- Larry
- Three Fund
- Sandwich
- Swensen
"""


TICKERS = dict()
WEIGHTS = dict()


TICKERS["All Weather"] = ["ITOT", "SCHQ", "SCHR", "GSP", "SGOL"]
WEIGHTS["All Weather"] = [0.3, 0.4, 0.15, 0.075, 0.075]

TICKERS["Butterfly"] = ["ITOT", "VBR", "SCHQ", "SCHO", "SGOL"]
WEIGHTS["Butterfly"] = [0.2, 0.2, 0.2, 0.2, 0.2]

TICKERS["60/40"] = ["ITOT", "SCHR"]
WEIGHTS["60/40"] = [0.6, 0.4]

TICKERS["Core Four"] = ["ITOT", "SPDW", "SCHR", "SCHH"]
WEIGHTS["Core Four"] = [0.48, 0.24, 0.2, 0.08]

TICKERS["Coffeehouse"] = ["SCHX", "SCHV", "SCHA", "VBR", "SPDW", "SCHR", "SCHH"]
WEIGHTS["Coffeehouse"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]

TICKERS["Global"] = ["SPTM", "SPDW", "VWO", "SPTI", "BWX", "SCHH", "SGOL"]
WEIGHTS["Global"] = [0.225, 0.225, 0.05, 0.176, 0.264, 0.04, 0.02]

TICKERS["Ideal"] = ["SCHX", "SCHV", "VBR", "VBK", "SPDW", "SCHO", "SCHH"]
WEIGHTS["Ideal"] = [0.0625, 0.0925, 0.0625, 0.0925, 0.31, 0.30, 0.08]

TICKERS["Larry"] = ["VBR", "ISVL", "VWO", "SCHR"]
WEIGHTS["Larry"] = [0.15, 0.075, 0.075, 0.7]

TICKERS["Three Fund"] = ["ITOT", "SPDW", "SCHR"]
WEIGHTS["Three Fund"] = [0.48, 0.12, 0.4]

TICKERS["Sandwich"] = [
    "SCHX",
    "SCHA",
    "SPDW",
    "SCHC",
    "VWO",
    "SCHR",
    "BIL",
    "IGOV",
    "SCHH",
]
WEIGHTS["Sandwich"] = [0.2, 0.08, 0.06, 0.1, 0.06, 0.3, 0.11, 0.04, 0.05]

TICKERS["Swensen"] = ["ITOT", "SPDW", "VWO", "SCHR", "SCHH"]
WEIGHTS["Swensen"] = [0.3, 0.15, 0.05, 0.3, 0.2]
