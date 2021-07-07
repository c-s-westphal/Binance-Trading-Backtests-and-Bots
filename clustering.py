# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:35:12 2021

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

client = Client('Key', 'Secret Key',{"verify": False, "timeout": 20})

def symbols():

    sym = ['BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','VET','GRT','DOGE','COMP','CHZ','LINK','SNX','YFI','CAKE','DOT','FIO','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
    
    #sym = ['BTC','ETH']
    
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
        



    data = [close, opens]
    
    
    df = pd.DataFrame (data)
    df = df.T
    df.columns = [ 'Closing price', 'Opening price']
   
    
    return (df)

def df2(symbol, datefrom):
    
    df = df_generator_15min(symbol, datefrom)
    
    df.dropna(inplace=True)
    
    df['Returns'] = np.log(df['Closing price']/df['Closing price'].shift(1))
    
    df.dropna(inplace=True)
    
    df['Direction'] = np.sign(df['Returns']).astype(int)
    
    #df['Returns'].hist(bins=35, figsize=(10,6))
    
    #df['Returns'].cumsum().apply(np.exp).plot()
    
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

def cluster_pred(symbol, datefrom):
    df = df2(symbol, datefrom)
    df = create_lags(df)
    
    model = KMeans(n_clusters=2, random_state=0)
    
    look_back = 10000
    
    predictions = []
    
    for i in range(look_back, len(df)):
        predictions.append(model.fit(df[cols][i-look_back:i]).predict(df[cols][i:i+1]))
       
    df = df[look_back:len(df)]
    
    predictions = np.array(predictions)
    
    df['pos_clus'] = np.where(predictions == 1, -1, 1)
    
    '''    for i in range(100,len(df)-1):
        if predictions[i] > stat.mean(df['Returns'][i-100:i-1]) :
            df['pos_clus'][i:i+1] = 1
        if predictions[i] < stat.mean(df['Returns'][i-100:i-1]) :
            df['pos_clus'][i:i+1] = -1
        else:
            df['pos_clus'][i:i+1] = 1'''
        
    df['trades'] = df['pos_clus'].diff()
    df['fees'] = np.where(df['trades'] != 0, (0.9**2), 1)
    
    df['strat_clus'] = df['pos_clus']*((df['Returns']))*df['fees']
    
    df[['Returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10,6))
    
    return df, predictions

avg_ret = []
avg_ret1 = []   
symbol = symbols()
for i in range(len(symbol)):
    df, j = (cluster_pred(symbol[i], '01 Dec 2020'))
    print(j)
    print(df)
    avg_ret.append(sum(df['Returns']))
    avg_ret1.append(sum(df['strat_clus']))
