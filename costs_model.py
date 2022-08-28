import numpy as np
import pandas as pd


class CostsModel():
    def __init__(self, asset_price, asset_number) -> None:
        self.asset_price = asset_price
        self.asset_number = asset_number
        self.trade_value = np.dot(self.asset_price, self.asset_number)
        #?taxation on trade value vs net profit

        # pass the list of security pdr objects, use get_quote to get details such as
        # exchanges, markets, individual trade sizes etc
        # pass those to the functions themselves

    def fees(self, market):
        """
        Fees (comission, fees and duties) for Interactive Brokers for Stocks and ETFs.
        Tiered pricing.

        Args:
            market (_type_): _description_

        Raises:
            ValueError: Error when calculation for an unavailable market is entered.

        Returns:
            float: Total fee
        """

        if market=="us": # currency is USD
            comission = np.min(1*self.trade_value, np.max(0.35, self.asset_number*0.0035))
            regulatory_total = 0.0000229*self.trade_value+0.00013*self.asset_number
            exchange = None    #TODO
            clearing = 0.00020*self.asset_number
            pass_through_total = 0.000175*comission+np.min(6.49, 0.00056*comission)
            total_fee = comission+regulatory_total+exchange+clearing+pass_through_total
        elif market=="sg": # currency is SGD
            comission = np.max(2.5, 0.0008*self.trade_value)
            exchange = (0.00034775+0.00008025)*self.trade_value
            total_fee = comission+exchange
        elif market=="hk": # currency is HKD
            comission = np.max(18, 0.0005*self.trade_value)
            exchange = 0.5*self.asset_price.shape[0]+0.00005*self.trade_value
            for i in range(self.asset_price.shape[0]):
                clearing += np.min(100, np.max(2, 0.00002*self.asset_price[i]*self.asset_number[i]))
            regulatory_total = (0.000027 + 0.0000015)*self.trade_value + np.round(0.0013*self.trade_value)
            total_fee = comission+exchange+clearing+regulatory_total
        elif market=="ldn": # curency is GBP
            comission = np.max(1, self.trade_value*0.0005)
            exchange = np.max(0.1, self.trade_value*0.000045)    # for LSE, LSEETF, LSEIOB1
            clearing = 0.06
            stamp_duty = 0.005*self.trade_value
            total_fee = comission+exchange+clearing+stamp_duty
        else:
            raise ValueError("Market unavailable.")

        return total_fee

    def tax(self, residence, accumulating, domiciled):
        if residence=="sg":
            if accumulating is True:
                total_tax=0
            else:
                if domiciled=="ireland":
                    total_tax=self.trade_value*0.15
                elif domiciled=="us"
                    total_tax=self.trade_value*0.3
        elif residence=="us":
            total_tax=self._us_capital_gains()
        else:
            raise ValueError("Costs for the given tax residence unavailable.")

        return total_tax

        def _us_capital_gains(self):
            pass
