"""
`portfolios.py` contains dictionaries with tickers and weights of assets for some notable portfolio examples.

Portfolios included:
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
- Comprehensive

Portfolio examples are mostly taken from [PortfolioCharts](portfoliocharts.com/portfolios/).
"""

TICKERS = dict()
WEIGHTS = dict()

# All Weather portfolio by Ray Dalio as per portfoliocharts.com (https://portfoliocharts.com/portfolios/all-seasons-portfolio/)
TICKERS["All Weather"] = ["ITOT", "SCHQ", "SCHR", "GSP", "SGOL"]
WEIGHTS["All Weather"] = [0.3, 0.4, 0.15, 0.075, 0.075]

# Golden Butterfly Portfolio by Tyler as per portfoliocharts.com (https://portfoliocharts.com/portfolios/golden-butterfly-portfolio/)
TICKERS["Butterfly"] = ["ITOT", "VBR", "SCHQ", "SCHO", "SGOL"]
WEIGHTS["Butterfly"] = [0.2, 0.2, 0.2, 0.2, 0.2]

# Classic 60-40 portfolio by John Bogle as per portfoliocharts.com (https://portfoliocharts.com/portfolios/classic-60-40-portfolio/)
TICKERS["60/40"] = ["ITOT", "SCHR"]
WEIGHTS["60/40"] = [0.6, 0.4]

# Core Four Portfolio by Rick Ferri as per portfoliocharts.com (https://portfoliocharts.com/portfolios/core-four-portfolio/)
TICKERS["Core Four"] = ["ITOT", "SPDW", "SCHR", "SCHH"]
WEIGHTS["Core Four"] = [0.48, 0.24, 0.2, 0.08]

# Coffeehouse Portfolio by Bill Schultheis as per portfoliocharts.com (https://portfoliocharts.com/portfolios/coffeehouse-portfolio/)
TICKERS["Coffeehouse"] = ["SCHX", "SCHV", "SCHA", "VBR", "SPDW", "SCHR", "SCHH"]
WEIGHTS["Coffeehouse"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]

# The Global Market Portfolio by Doeswijk, Lam, and Swinkels as per portfoliocharts.com (https://portfoliocharts.com/portfolios/global-market-portfolio/)
TICKERS["Global"] = ["SPTM", "SPDW", "VWO", "SPTI", "BWX", "SCHH", "SGOL"]
WEIGHTS["Global"] = [0.225, 0.225, 0.05, 0.176, 0.264, 0.04, 0.02]

# Ideal Index Portfolio by Frank Armstrong as per portfolio charts (https://portfoliocharts.com/portfolios/ideal-index-portfolio/)
TICKERS["Ideal"] = ["SCHX", "SCHV", "VBR", "VBK", "SPDW", "SCHO", "SCHH"]
WEIGHTS["Ideal"] = [0.0625, 0.0925, 0.0625, 0.0925, 0.31, 0.30, 0.08]

# Larry Portfolio by Larry Swedroe as per portfoliocharts.com (https://portfoliocharts.com/portfolios/larry-portfolio/)
TICKERS["Larry"] = ["VBR", "ISVL", "VWO", "SCHR"]
WEIGHTS["Larry"] = [0.15, 0.075, 0.075, 0.7]

# Three-fund portfolio as per portfoliocharts.com (https://portfoliocharts.com/portfolios/three-fund-portfolio/)
TICKERS["Three Fund"] = ["ITOT", "SPDW", "SCHR"]
WEIGHTS["Three Fund"] = [0.48, 0.12, 0.4]

# Sandwich Portfolio by Bob Clyatt as per portfoliocharts.com (https://portfoliocharts.com/portfolios/sandwich-portfolio/)
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

# David Swensen's portfolio as per portfoliocharts.com (https://portfoliocharts.com/portfolios/swensen-portfolio/)
TICKERS["Swensen"] = ["ITOT", "SPDW", "VWO", "SCHR", "SCHH"]
WEIGHTS["Swensen"] = [0.3, 0.15, 0.05, 0.3, 0.2]

# Comprehensive portfolio from "The Risk and Reward of Investing" by Ronald Doeswijk and Laurens Swinkels (2024)
TICKERS["Comprehensive"] = ["VT", "REET", "BND", "DBC"]
WEIGHTS["Comprehensive"] = [0.493, 0.045, 0.444, 0.018]
