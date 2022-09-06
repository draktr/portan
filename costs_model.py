import numpy as np
import pandas as pd
import pandas_datareader as pdr


class CostsModel():

    def __init__(self,
                 tickers,
                 asset_number,
                 bought,
                 sold) -> None:

        if np.dot(bought, sold) != 0:
            raise ValueError("Securities can either be only bought or only sold.")

        data = pdr.get_quote_yahoo(tickers)
        current_prices = data["regularMarketPrice"]

        self.trades = pd.DataFrame({"tickers":tickers,
                                    "asset_number":asset_number,
                                    "bought":bought,
                                    "sold":sold,
                                    "exchange": data["fullExchangeName"],
                                    "transaction_value": np.multiply(asset_number, current_prices),
                                    "total_fees":np.array(len(asset_number))})

    def total_fees(self, brokerage="ib_fixed"):

        if brokerage == "ib_fixed":
            for transaction in range(len(self.trades)):
                self.trades.loc[transaction, "total_fees"] = self._fees_ib_fixed(self.trades.loc[transaction])
        else:
            raise ValueError("Brokerage unavailable.")

    def _fees_ib_fixed(self, transaction):

        if transaction["exchange"] == "NasdaqGS" or "NYSE":        # currency is USD
            comission = np.min(0.01*transaction["transaction_value"], np.max(1, transaction["asset_number"]*0.005))
            regulatory_total = 0.0000229*transaction["transaction_value"]+0.00013*transaction["asset_number"]
            total_fee = comission+regulatory_total
        elif transaction["exchange"] == "SES":                     # currency is SGD
            comission = np.max(2.5, 0.0008*transaction["transaction_value"])
            total_fee = comission
        elif transaction["exchange"] == "HKSE":                    # currency is HKD
            comission = np.max(18, 0.0008*transaction["transaction_value"])
            regulatory_total = (0.000027 + 0.0000015)*transaction["transaction_value"] + np.round(0.0013*transaction["transaction_value"])
            total_fee = comission+regulatory_total
        elif transaction["exchange"] == "LSE":                     # curency is GBP
            comission = np.max(3, transaction["transaction_value"]*0.0005)
            total_fee = comission
        else:
            raise ValueError("Market unavailable.")

        return total_fee


def main():
    tickers = ["AAPL", "XOM", "META"]
    asset_number = [4, 7, 3]
    bought = [0, 1, 0]
    sold = [1, 0, 1]

    costs = CostsModel(tickers, asset_number, bought, sold)
    costs.total_fees()

    return costs.trades

if __name__=="__main__":
    main()
