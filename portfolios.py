import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf


class Portfolios():

    def __init__(self) -> None:
        pass

    def get_singlife(self):
        # Singlife Sure Invest Dynamic Portfolio as of June 30th 2022
        tickers = ["0P00006G1V.SI", "0P0000SO1U.SI", "0P0000ZJWQ.SI", "0P00016FYC.SI", "0P0001CB2H.SI", "0P0001FL3C.SI", "0P00006HYS.SI", "0P00018FHU.SI", "0P0001BONF.SI", "0P0001OK4Y.SI"]
        weights = [0.2004, 0.02, 0.999, 0.1199, 0.01, 0.0601, 0.2098, 0.1299, 0.1098, 0.0402]

        singlife_dynamic_t = pd.DataFrame(tickers, columns=["Singlife Dynamic 30/6/2022"])
        singlife_dynamic_w = pd.DataFrame(weights, columns=["Singlife Dynamic 30/6/2022"])

        return singlife_dynamic_t, singlife_dynamic_w


    def get_classics(self):

        all_weather_t = ["ITOT", "SCHQ", "SCHR", "GSP", "SGOL"]
        all_weather_w = [0.3, 0.4, 0.15, 0.075, 0.075]

        butterfly_t = ["ITOT", "VBR", "SCHQ", "SCHO", "SGOL"]
        butterfly_w = [0.2, 0.2, 0.2, 0.2, 0.2]

        sixty_fourty_t = ["ITOT", "SCHR"]
        sixty_fourty_w = [0.6, 0.4]

        core_four_t = ["ITOT", "SPDW", "SCHR", "SCHH"]
        core_four_w = [0.48, 0.24, 0.2, 0.08]

        #coffeehouse_t = ["SCHX", "SCHV", "SCHA", "VBR", "SPDW", "SCHR", "SCHH"]
        #coffeehouse_w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]

        global_t = ["SPTM", "SPDW", "VWO", "SPTI", "BWX", "SCHH", "SGOL"]
        global_w = [0.225, 0.225, 0.05, 0.176, 0.264, 0.04, 0.02]

        ideal_t = ["SCHX", "SCHV", "VBR", "VBK", "SPDW", "SCHO", "SCHH"]
        ideal_w = [0.0625, 0.0925, 0.0625, 0.0925, 0.31, 0.30, 0.08]

        #larry_t = ["VBR", "ISVL", "VWO", "SCHR"]
        #larry_w = [0.15, 0.075, 0.075, 0.7]

        three_fund_t = ["ITOT", "SPDW", "SCHR"]
        three_fund_w = [0.48, 0.12, 0.4]

        #sandwich_t = ["SCHX", "SCHA", "SPDW", "SCHC", "VWO", "SCHR", "BIL", "IGOV", "SCHH"]
        #sandwich_w = [0.2, 0.08, 0.06, 0.1, 0.06, 0.3, 0.11, 0.04, 0.05]

        swensen_t = ["ITOT", "SPDW", "VWO", "SCHR", "SCHH"]
        swensen_w = [0.3, 0.15, 0.05, 0.3, 0.2]

        portfolios_t = pd.concat([pd.Series(sixty_fourty_t), pd.Series(all_weather_t),
                                  pd.Series(butterfly_t), pd.Series(core_four_t),
                                  pd.Series(global_t), pd.Series(ideal_t),
                                  pd.Series(swensen_t), pd.Series(three_fund_t)], axis=1)
        portfolios_t.columns = ["60-40", "All Weather", "Butterfly", "Core-4", "Global Market", "Ideal Index",
                                "Swensen", "Three Fund"]

        portfolios_w = pd.concat([pd.Series(sixty_fourty_w), pd.Series(all_weather_w),
                                  pd.Series(butterfly_w), pd.Series(core_four_w),
                                  pd.Series(global_w), pd.Series(ideal_w),
                                  pd.Series(swensen_w), pd.Series(three_fund_w)], axis=1)
        portfolios_w.columns = ["60-40", "All Weather", "Butterfly", "Core-4", "Global Market", "Ideal Index",
                                "Swensen", "Three Fund"]


        return portfolios_t, portfolios_w

    def single_portfolio_returns(self,
                                 assets_tickers,
                                 assets_weights,
                                 start="2010-01-01",
                                 end=str(datetime.now())[0:10],
                                 data=None,
                                 portfolio_name="Investment Portfolio"):

        if data is None:
            prices = pd.DataFrame(columns=assets_tickers)
            assets_returns = pd.DataFrame(columns=assets_tickers)

            for tick in assets_tickers:
                price_current = yf.download(tick, start=start, end=end) # TODO: check if can be done without the loop (with pdr)
                prices[tick] = price_current["Adj Close"]
                assets_returns[tick] = price_current["Adj Close"].pct_change()
        else:
            prices = pd.read_csv(data, index_col=["Date"])
            assets_returns = pd.DataFrame(columns=prices.columns, index=prices.index)
            assets_returns = prices.pct_change()
            start = prices.index[0]
            end = prices.index[-1]

        assets_returns = assets_returns.drop(assets_returns.index[0])
        portfolio_returns = np.dot(assets_returns.to_numpy(), assets_weights)
        portfolio_returns = pd.DataFrame(portfolio_returns,
                                         columns=[portfolio_name],
                                         index=assets_returns.index)

        return portfolio_returns

    def get_characteristics(self,
                            portfolios,
                            weights):

        mean_returns = pd.Series(index=portfolios.columns, dtype="float64")
        volatilities = pd.Series(index=portfolios.columns, dtype="float64")
        portfolios_returns = pd.DataFrame(columns=portfolios.columns, dtype="float64")

        for portfolio in portfolios.columns:
            portfolios_returns[portfolio]=self.single_portfolio_returns(portfolios[portfolio].dropna(),
                                                                        weights[portfolio].dropna(),
                                                                        start="2010-01-01",
                                                                        end=str(datetime.now())[0:10],
                                                                        data=None,
                                                                        portfolio_name=portfolio)
            mean_returns[portfolio] = np.nanmean(portfolios_returns[portfolio])
            volatilities[portfolio] = np.nanstd(portfolios_returns[portfolio])

        return mean_returns, volatilities
