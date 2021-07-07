# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:28:45 2021

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
from sklearn.cluster import KMeans

client = Client('Key', 'secret key',{"verify": False, "timeout": 20})

def symbols():

    sym = ['BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','VET','GRT','DOGE','COMP','CHZ','LINK','SNX','YFI','CAKE','DOT','FIO','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
    
    symbols = []
    
    for i in range(len(sym)):
        symbols.append(sym[i] + 'BUSD')
    
    return symbols


def df_generator_15min(symbol, datefrom):
 
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, datefrom)
    
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
    
    #print(klines)
    
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
        



    data = [opens, close, period_return, percentage_change, high, low, volume, open_time, close_time]
    
    
    df = pd.DataFrame (data)
    df = df.T
    df.columns = ['Opening price', 'Closing price', 'Period return','Percentage change', 'High', 'Low', 'Volume', 'Opening time','Closing time']
   
    
    return (df)

def df2(symbol, datefrom):
    
    df = df_generator_15min(symbol, datefrom)
    
    df.dropna(inplace=True)
    
    df['returns'] = np.log(df['Closing price']/df['Closing price'].shift(1))
    
    df.dropna(inplace=True)
    
    df['direction'] = np.sign(df['returns']).astype(int)
    
    #df['Returns'].hist(bins=35, figsize=(10,6))
    
    #df['Returns'].cumsum().apply(np.exp).plot()
    
    return df

def create_lags(df):
    global cols
    cols = []
    lags = 3
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['returns'].shift(lag)
        cols.append(col)
        
    df.dropna(inplace=True)
    
    return df

def create_bins(data, bins=[0] ):
    global cols_bin
    
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        cols_bin.append(col_bin)
        data[col_bin] = np.digitize(data[col], bins=bins)
        
    print(data[cols_bin])
        
    return data

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
    

def freq_bt(symbol, datefrom):
    
    df = df2(symbol, datefrom)
    df = create_lags(df)
    data = create_bins(df)
    
    #grouped = data.groupby(cols_bin + ['direction'])
    
    #res = grouped['direction'].size().unstack(fill_value=0)
    
    #res.style.apply(highlight_max, axis=1)
    
    data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 0, -1, 1)
    
    data['strat_freq'] = data['pos_freq']*data['returns']
    
    data[['returns','strat_freq']].cumsum().apply(np.exp).plot(figsize=(10,6))
    
    return data

print(freq_bt('BTCBUSD', '26 Mar 2021'))
