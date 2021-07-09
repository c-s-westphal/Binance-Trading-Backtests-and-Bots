# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:11:03 2021

@author: Charlie
@reviewer : MonkD3
"""

from binance.client import Client
import pandas as pd
import numpy as np
from binance.enums import *
from decimal import *
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time


class Linreg :

    def __init__(self, key, secret_key):
        self._client = Client('key', 'secret key',{"verify": False, "timeout": 200000})
        self._cols = []
        self._cols_bin = []
        self._global_symbols = ['VET', 'ADA', 'MATIC', 'ETH', '1INCH', 'XLM']


    def get_global_symbols(self):
        return self._global_symbols


    def symbols(self):

        sym = ['BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','VET','GRT','DOGE','COMP','CHZ','LINK','SNX','YFI','CAKE','DOT','FIO','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
        
        # use the built-in function "map" to be more efficient/elegant (see https://docs.python.org/3/library/functions.html#map) :
        symbols = list(map(lambda symbol : symbol + "BUSD", sym))

        return symbols


    def df_generator_15min(self, symbol, datefrom):
        
        cross = symbol + 'BUSD'
        klines = self._client.get_historical_klines(cross, Client.KLINE_INTERVAL_30MINUTE, datefrom)

        data = pd.DataFrame(klines[4], dtype=np.float64).T 
        data.columns = ["Closing price"]

        return data

    def calc_direction(self, symbol, datefrom):
        
        df = self.df_generator_15min(symbol, datefrom)
        df.dropna(inplace=True)
        
        df['Returns'] = np.log(df['Closing price']/df['Closing price'].shift(1))
        df.dropna(inplace=True)
        
        df['Direction'] = np.sign(df['Returns']).astype(int)
        
        return df
    
    def create_lags(self, df):
        lags = 1
        for lag in range(1, lags+1):
            col = 'lag_{}'.format(lag)
            df[col] = df['Returns'].shift(lag)
            self._cols.append(col)
            
        df.dropna(inplace=True)
        return df # You do not really need to return the result because you modified it right in the parameter you gave (see "reference passing")

    def create_bins(self, data, bins):
        for col in self._cols:
            col_bin = col + '_bin'
            self._cols_bin.append(col_bin)
            data[col_bin] = np.digitize(data[col], bins=bins)
            
        print(data[self._cols_bin])
            
        return data

    def regressor(self, df):
        
        model = LinearRegression()
        N = len(df)
        
        model.fit(df[self._cols],df['Direction'])
        latest = np.array([df['Returns'][N-1:N]])
                            
        X = model.predict(latest.T)
            
        #plt.scatter(df[self._cols], df['Direction'],  color='black')
        #plt.plot(latest, X, color='blue', linewidth=3)
    
        return X

    def date_from_gen():
        N = 208

        date_N_days_ago = datetime.now() - timedelta(days=N)
        j = str(date_N_days_ago)
        
        return j

    def model(self, symbol):
        
        datefrom = self.date_from_gen()

        # More readable way
        df = self.calc_direction(symbol, datefrom)
        pred = self.regressor(self.create_lags(df))

        return pred, df

    def _execute_buy_order(self, cross, qty):
        self._client.create_margin_order(symbol=cross, side = SIDE_BUY, 
                                        type = 'MARKET', sideEffectType = 'AUTO_REPAY', 
                                        quantity = qty)    
        self._client.create_margin_order(symbol=cross, side = SIDE_BUY, 
                                        type = 'MARKET',  quantity= qty)

    def _execute_sell_order(self, symbol, cross, qty):
        self._client.create_margin_order(symbol=cross,  side = SIDE_SELL, 
                                        type = 'MARKET',  quantity=(qty))

        self._client.create_margin_loan(asset= symbol, amount=qty)

        self._client.create_margin_order(symbol=cross,  side = SIDE_SELL, 
                                        type = 'MARKET',  quantity=(qty))

    def order(self, symbol):
        cross = symbol + 'USDT'
        pred, df = self.model(symbol)
        N = len(df)
        
        acct = self._client.get_margin_account()
        orders = self._client.get_all_margin_orders(symbol=cross, limit=1)
        order_p = float(orders[0]['cummulativeQuoteQty'])/float(orders[0]['origQty'])
        
        candles = self._client.get_klines(symbol=cross, interval=Client.KLINE_INTERVAL_5MINUTE)
        N = len(candles)
        open_p = float(candles[N-1][1])
        close_p = float(candles[N-1][4])
        
        for i in range(len(acct['userAssets'])):
            if pred >= 0 and (close_p - open_p) > 0 and acct['userAssets'][i]['asset'] == symbol and float(acct['userAssets'][i]['borrowed']) > 11/close_p and (close_p - open_p) > 0 and (close_p - order_p) < 0:
            
                desired = float(acct['userAssets'][i]['borrowed'])
                info = self._client.get_symbol_info(symbol=cross)
                minimum = float(info['filters'][2]['minQty'])
                stepsize = float(info['filters'][2]['stepSize'])
                val = desired/stepsize
                quantity_bs = (minimum*int(val))
                while quantity_bs < desired:
                    quantity_bs = quantity_bs + minimum
                
                while quantity_bs > desired:
                    quantity_bs = quantity_bs - minimum
                    
                self._execute_buy_order(cross, quantity_bs)
                
            
            if acct['userAssets'][i]['asset'] == symbol and float(acct['userAssets'][i]['borrowed']) > 11/close_p and (close_p - order_p) > 0:
            
                desired = float(acct['userAssets'][i]['borrowed'])
                info = self._client.get_symbol_info(symbol=cross)
                minimum = float(info['filters'][2]['minQty'])
                stepsize = float(info['filters'][2]['stepSize'])
                val = desired/stepsize
                quantity_bs = (minimum*int(val))
                while quantity_bs < desired:
                    quantity_bs = quantity_bs + minimum
                
                while quantity_bs > desired:
                    quantity_bs = quantity_bs - minimum
                    
                self._execute_buy_order(cross, quantity_bs)
            
        
            if pred < 0 and (close_p - open_p) < 0 and acct['userAssets'][i]['asset'] == symbol and float(acct['userAssets'][i]['free']) > 1:
                desired = float(acct['userAssets'][i]['free'])
                info = self._client.get_symbol_info(symbol=cross)
                minimum = float(info['filters'][2]['minQty'])
                stepsize = float(info['filters'][2]['stepSize'])
                val = desired/stepsize
                quantity_s = (minimum*int(val))
                while quantity_s < desired:
                    quantity_s = quantity_s + minimum
                    
                while quantity_s > desired:
                    quantity_s = quantity_s - minimum
                
                self._execute_sell_order(symbol, cross, quantity_s)

        print(pred)


    def __call__(self, symbols):
        for symbol in symbols :
            self.order(symbol)
    

# BIG WARNING : you can't import this file into another, 
#               if you do this the infinite loop will run and you won't
#               understand why the another file doesn't work.
# So if you want to import this file into another, just put the loop
# into a function or delete it

runner = Linreg("key", "secret key")
while True :
    runner(runner.get_global_symbols()) # global_symbols is a global variable defined in the beginning of the file
    time.sleep(222)