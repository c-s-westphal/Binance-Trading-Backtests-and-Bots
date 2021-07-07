# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:11:03 2021

@author: Charlie
"""

from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from binance.enums import *
from csv import writer
from decimal import *
import datetime
import xlsxwriter
import csv
from functools import reduce  # Required in Python 3
import operator
import scipy.stats as ss
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import schedule
import time

client = Client('key', 'secret key',{"verify": False, "timeout": 200000})

def symbols():

    sym = ['BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','VET','GRT','DOGE','COMP','CHZ','LINK','SNX','YFI','CAKE','DOT','FIO','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
    
    symbols = []
    
    for i in range(len(sym)):
        symbols.append(sym[i] + 'BUSD')
    
    return symbols


def df_generator_15min(symbol, datefrom):
    
    cross = symbol + 'BUSD'
 
    klines = client.get_historical_klines(cross, Client.KLINE_INTERVAL_30MINUTE, datefrom)
    
    open_time = []
    opens = []
    high = []
    low = []
    close = []
    volume = []
    close_time = []
    quote_asset_volume = []
    number_of_trades = []
    taker_buy_base_asset_volume = []
    taker_buy_quote_asset_volume = []  
    period_return = []
    percentage_change = []

    
    for i in range(len(klines)):
        open_time.append(float(klines[i][0]))
        opens.append(float(klines[i][1]))
        high.append(float(klines[i][2]))
        low.append(float(klines[i][3]))
        close.append(float(klines[i][4]))
        volume.append(float(klines[i][5]))
        close_time.append(float(klines[i][6]))
        quote_asset_volume.append(float(klines[i][7]))
        number_of_trades.append(float(klines[i][8]))
        taker_buy_base_asset_volume.append(float(klines[i][9]))
        taker_buy_quote_asset_volume.append(float(klines[i][10]))
        period_return.append(float(klines[i][4])-float(klines[i][1]))
        percent = (float(klines[i][4])-float(klines[i][1]))/(float(klines[i][1]))
        percentage_change.append(percent)
        



    data = [close]
    
    
    df = pd.DataFrame (data)
    df = df.T
    df.columns = [ 'Closing price']
   
    
    return (df)

def df2(symbol, datefrom):
    
    df = df_generator_15min(symbol, datefrom)
    
    df.dropna(inplace=True)
    
    df['Returns'] = np.log(df['Closing price']/df['Closing price'].shift(1))
    
    df.dropna(inplace=True)
    
    df['Direction'] = np.sign(df['Returns']).astype(int)
    
    return df
    
def create_lags(df):
    global cols
    cols = []
    lags = 1
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['Returns'].shift(lag)
        cols.append(col)
        
    df.dropna(inplace=True)
    
    return df

def create_bins(data, bins):
    global cols_bin
    
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        cols_bin.append(col_bin)
        data[col_bin] = np.digitize(data[col], bins=bins)
        
    print(data[cols_bin])
        
    return data

def regressor(df):
      
    
    
    model = LinearRegression()
    
    N = len(df)
    
    
    model.fit(df[cols],df['Direction'])
                         
    latest = np.array([df['Returns'][N-1:N]])
                         
    X = model.predict(latest.T)
        
    #plt.scatter(df[cols], df['Direction'],  color='black')
    #plt.plot(latest, X, color='blue', linewidth=3)
   
    return X

def date_from_gen():
    

    N = 208

    date_N_days_ago = datetime.now() - timedelta(days=N)
    j = str(date_N_days_ago)
    
    return j

def model(symbol):
    
    datefrom = date_from_gen()
    
    df = df2(symbol, datefrom)
    
    df = create_lags(df)
    
    pred = regressor(df)
    
    return pred, df

def orders(symbol):
    
    pred, df = model(symbol)
    
    N = len(df)
    
    cross = symbol + 'USDT'
    
    
    acct = client.get_margin_account()
    orders = client.get_all_margin_orders(symbol=cross, limit=1)
    order_p = float(orders[0]['cummulativeQuoteQty'])/float(orders[0]['origQty'])
    
    candles = client.get_klines(symbol=cross, interval=Client.KLINE_INTERVAL_5MINUTE)
    N = len(candles)
    open_p = float(candles[N-1][1])
    close_p = float(candles[N-1][4])
    
    
    
    
    for i in range(len(acct['userAssets'])):
        if pred >= 0 and (close_p - open_p) > 0 and acct['userAssets'][i]['asset'] == symbol and float(acct['userAssets'][i]['borrowed']) > 11/close_p and (close_p - open_p) > 0 and (close_p - order_p) < 0:
        
            desired = float(acct['userAssets'][i]['borrowed'])
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            stepsize = float(info['filters'][2]['stepSize'])
            val = desired/stepsize
            quantity_bs = (minimum*int(val))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
            
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
                
        
        
      
            
            client.create_margin_order(symbol=cross,  side = SIDE_BUY, type = 'MARKET',  sideEffectType = 'AUTO_REPAY', quantity = quantity_bs )
        
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_bs )
        
        if acct['userAssets'][i]['asset'] == 'VET' and float(acct['userAssets'][i]['borrowed']) > 11/close_p and (close_p - order_p) > 0:
        
            desired = float(acct['userAssets'][i]['borrowed'])
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            stepsize = float(info['filters'][2]['stepSize'])
            val = desired/stepsize
            quantity_bs = (minimum*int(val))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
            
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
                
        
        
      
            
            client.create_margin_order(symbol=cross,  side = SIDE_BUY, type = 'MARKET',  sideEffectType = 'AUTO_REPAY', quantity = quantity_bs )
        
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_bs )
        
       
        if pred < 0 and (close_p - open_p) < 0 and acct['userAssets'][i]['asset'] == symbol and float(acct['userAssets'][i]['free']) > 1:
            desired = float(acct['userAssets'][i]['free'])
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            stepsize = float(info['filters'][2]['stepSize'])
            val = desired/stepsize
            quantity_s = (minimum*int(val))
            while quantity_s < desired:
                quantity_s = quantity_s + minimum
                
            while quantity_s > desired:
                quantity_s = quantity_s - minimum
            
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))
            client.create_margin_loan(asset=symbol, amount=quantity_s)
        
        
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))


    return print(pred)

def orders_across_symbols():
    
    symbols = ['ADA', 'MATIC', 'BTC', 'ETH', 'XLM']
    
    for i in range(len(symbols)):
        
        print(orders(symbols[i]))
              
    return


while '1' == '1':
    
    print(orders_across_symbols())
    
    time.sleep(222)
    
    



    
