# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:18:29 2021

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
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

client = Client('Key', 'key',{"verify": False, "timeout": 20})

def symbols():

    sym = ['BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','VET','GRT','DOGE','COMP','CHZ','LINK','SNX','YFI','CAKE','DOT','FIO','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
    
    symbols = []
    
    for i in range(len(sym)):
        symbols.append(sym[i] + 'BUSD')
    
    return symbols


def df_generator_15min(symbol, datefrom):
 
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, datefrom)
    
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

def create_bins(data, bins):
    global cols_bin
    
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        cols_bin.append(col_bin)
        data[col_bin] = np.digitize(data[col], bins=bins)
        
    print(data[cols_bin])
        
    return data

def svc_pred(symbol, datefrom):
    df = df2(symbol, datefrom)
    df = create_lags(df)
  
    
    
    C=1
    model = SVC(C=C)
    
    look_back = 10000
    
    predictions = []
    
    for i in range(look_back, len(df)):
        mu = df['Returns'][i-look_back:i].mean()
        v = df['Returns'][i-look_back:i].std()
        bins = [mu - v, mu, mu + v]
        df = create_bins(df, bins)
        predictions.append(model.fit(df[cols_bin][i-look_back:i],df['Direction'][i-look_back:i]).predict(df[cols_bin][i:i+1]))
       
    df = df[look_back:len(df)]
    
    predictions = np.array(predictions)
    
    df['pos_clus'] = np.where(predictions == -1, -1, 1)
    
    '''    for i in range(100,len(df)-1):
        if predictions[i] > stat.mean(df['Returns'][i-100:i-1]) :
            df['pos_clus'][i:i+1] = 1
        if predictions[i] < stat.mean(df['Returns'][i-100:i-1]) :
            df['pos_clus'][i:i+1] = -1
        else:
            df['pos_clus'][i:i+1] = 1'''
        
    df['trades'] = df['pos_clus'].diff()
    df['fees'] = np.where(df['trades'] != 0, (0.925**2), 1)
    
    df['strat_clus'] = df['pos_clus']*df['Returns']*df['fees']
    
    df[['Returns', 'strat_clus']].cumsum().plot(figsize=(10,6))
    
    return df, predictions
   
    
avg_ret = []
avg_ret1 = []   
symbol = symbols()
for i in range(len(symbol)):
    df, j = (svc_pred(symbol[i], '01 Oct 2020'))
    print(j)
    print(df)
    avg_ret.append(sum(df['Returns']))
    avg_ret1.append(sum(df['strat_clus']))
    
print(avg_ret, avg_ret1)
print(sum(avg_ret), sum(avg_ret1))

#print(svc_pred('BTCBUSD','01 Oct 2020'))
