# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:00:51 2021

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
from sklearn.pipeline import make_pipeline

client = Client('Key', 'Binance private Key',{"verify": False, "timeout": 20})

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

def regressor(df):
      
    predictions_1 = []
    predictions_2 = []
    
    model = LinearRegression()
    
    look_back = 10000
    
    for i in range(look_back, len(df)):
        predictions_1.append(model.fit(df[cols][i-look_back:i],df['Returns'][i-look_back:i]).predict(df[cols][i:i+1]))
        predictions_2.append(model.fit(df[cols][i-look_back:i],df['Direction'][i-look_back:i]).predict(df[cols][i:i+1]))
        
    
    #print(predictions_1)
    #print(predictions_2)
    
    
    
    df = df[look_back:len(df)]
    
    predictions_1 = np.array(predictions_1)
    predictions_2 = np.array(predictions_2)
    
    #df['pos_ols_1'] = predictions_1
    #df['pos_ols_2'] = predictions_2
    
    df['pos_ols_1'] = np.where(predictions_1 > 0, 1, -1)
    df['pos_ols_2'] = np.where(predictions_2 > 0, 1, -1)
    
    '''for i in range(1,len(df)-1):
        if df['Returns'][i-1:i] < -0.08 :
            df['pos_ols_1'][i:i+1] = -1
        else:
            df['pos_ols_1'][i:i+1] = 1'''
        
    
   
    df['1_trades'] = df['pos_ols_1'].diff()
    df['2_trades'] = df['pos_ols_2'].diff()
    
    df['1_fees'] = np.where(df['1_trades'] != 0, (0.87**2), 1)
    df['2_fees'] = np.where(df['2_trades'] != 0, (0.87**2), 1)
    
    df['strat_ols_1'] = df['pos_ols_1']*df['Returns']*df['1_fees']
    df['strat_ols_2'] = df['pos_ols_2']*df['Returns']*df['2_fees']
    
    df[['Returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(15,6))
    
    ols_1_no_trade = (df['pos_ols_1'].diff() != 0).sum() 
    ols_2_no_trade = (df['pos_ols_2'].diff() != 0).sum() 
   
    return df, print(predictions_2)


#print(data2['Returns'][0:1])

'''model = LinearRegression()
#for i in range(50, len(data2)-49):
print(data2[cols],data2[cols][0:1])
print(data['Returns'][0:1])
print(model.fit(data2[cols][2:3],data2['Direction'][2:3]).predict(data2[cols][3:4]))
print(model.fit(data2[cols][2:3],data2['Returns'][2:3]).predict(data2[cols][3:4]))
'''

#print(j['pos_ols_1'][0:1])
#data=(df2('BTCBNB', '2 Jan 2021'))
#data2 = (create_lags(data))
#print(regressor(data2))

symbol = symbols()

avg_ret1 = []
avg_ret2 = []
avg_ret = []

for i in range(len(symbol)):
    data=(df2(symbol[i], '01 Jan 2021'))
    data2 = (create_lags(data))
    df, j = (regressor(data2))
    avg_ret.append(sum(df['Returns']))
    avg_ret1.append(sum(df['strat_ols_1']))
    avg_ret2.append(sum(df['strat_ols_2']))
    
print(avg_ret, avg_ret1, avg_ret2)
print(sum(avg_ret), sum(avg_ret1), sum(avg_ret2))
    
'''j=((df2('BTCBUSD', '26 Jan 2021')))
k = create_lags(j)

N = len(j)
df=j
l = np.array([df['Returns'][N-1:N],df['lag_1'][N-1:N],df['lag_2'][N-1:N],df['lag_3'][N-1:N],df['lag_4'][N-1:N],df['lag_5'][N-1:N]])

print(l.T)'''
