from curses.ascii import NUL
from multiprocessing.sharedctypes import Value
from turtle import clear
import numpy as np
import pandas as pd


class CostsModel():
    def __init__(self) -> None:
        pass
        # pass the list of security pdr objects, use get_quote to get details such as
        # exchanges, markets, individual trade sizes etc
        # pass those to the functions themselves

    def interactive_brokers_tiered_stock_etf(self, asset_price, asset_number, market):

        trade_value = np.dot(asset_price, asset_number)

        if market=="us": # currency is USD
            comission = np.min(1*trade_value, np.max(0.35, asset_number*0.0035))
            regulatory_total = 0.0000229*trade_value+0.00013*asset_number
            exchange = None    #TODO
            clearing = 0.00020*asset_number
            pass_through_total = 0.000175*comission+np.min(6.49, 0.00056*comission)
            total_fee = comission+regulatory_total+exchange+clearing+pass_through_total
        elif market=="sg": # currency is SGD
            comission = np.max(2.5, 0.0008*trade_value)
            exchange = (0.00034775+0.00008025)*trade_value
            total_fee = comission+exchange
        elif market=="hk": # currency is HKD
            comission = np.max(18, 0.0005*trade_value)
            exchange = 0.5*asset_price.shape[0]+0.00005*trade_value
            for i in range(asset_price.shape[0]):
                clearing += np.min(100, np.max(2, 0.00002*asset_price[i]*asset_number[i]))
            regulatory_total = (0.000027 + 0.0000015)*trade_value + np.round(0.0013*trade_value)
            total_fee = comission+exchange+clearing+regulatory_total
        elif market=="ldn": # curency is GBP
            comission = np.max(1, trade_value*0.0005)
            exchange = np.max(0.1, trade_value*0.000045)    # for LSE, LSEETF, LSEIOB1
            clearing = 0.06
            stamp_tax = 0.005*trade_value
            total_fee = comission+exchange+clearing+stamp_tax
        else:
            raise ValueError("Market unavailable.")


        return total_fee
