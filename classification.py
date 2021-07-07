# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:01:30 2021

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

client = Client('Key', 'secret Key',{"verify": False, "timeout": 20})

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
    lags = 2
    for lag in range(1, lags+1):
        col = 'lag_{}'.format(lag)
        df[col] = df['returns'].shift(lag)
        cols.append(col)
        
    df.dropna(inplace=True)
    
    return df

def create_bins(data, bins=[0]):
    global cols_bin 
    cols_bin = []
    for col in cols:
        col_bin = col +'bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)
            
    return data

def fit_models(data):
    mfit = {model: models[model].fit(data[cols_bin],
                                     data['direction']) 
            for model in models.keys()}
    
    return
    
def derive_positions(data):
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])
        
    return data
        
def evaluate_strats(data):
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['Returns']
        sel.append(col)
        
    sel.insert(0, 'Returns')
    
    data[sel].cumsum().apply(np.exp).plot(figsize=(10,6))
    
    return data
  
C = 1
models = {
    'log_reg': linear_model.LogisticRegression(C=C),
    'gauss_nb': GaussianNB(),
    'svm': SVC(C=C)
    }  

df=(df2('BTCBUSD', '26 Mar 2020'))
data = (create_lags(df))
data = create_bins(data)
data = fit_models(data)
data = derive_positions(data)   
data = evaluate_strats(data)
