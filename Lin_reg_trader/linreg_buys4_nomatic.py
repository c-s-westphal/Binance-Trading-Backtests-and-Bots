# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:25:00 2021

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
import math

client = Client('key', 'Secret Key',{"verify": False, "timeout": 200})

def symbols():

    sym = [ 'BTC', 'ETH', 'XRP','BCH','ADA','BAT','MATIC','GRT','COMP','LINK','SNX','YFI','CAKE','DOT','MKR','BNB','ZEC','EGLD','ZIL','EOS','LTC','XLM','XTZ','ETC']
    
    
    
    acct = client.get_margin_account()
    
    
    symbols = []
    
    #for i in range(len(acct['userAssets'])):
     #   symbols.append(acct['userAssets'][i]['asset'])
    
    return sym


def df_generator_15min(symbol, datefrom):
    
    cross = symbol + 'USDT'
 
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

def regressor(df):
      
    
    
    model = LinearRegression()
    
    N = len(df)
    
    
    model.fit(df[cols],df['Direction'])
                         
    latest = np.array([df['Returns'][N-1:N]])
                         
    X = model.predict(latest.T)
    
    plt.scatter(df[cols], df['Direction'],  color='black')
    plt.plot(latest, X, color='blue', linewidth=3)
   
    return X

def date_from_gen():
    

    N = 58

    date_N_days_ago = datetime.now() - timedelta(days=N)
    j = str(date_N_days_ago)
    
    return j

def date_from_1():
    

    N = 1

    date_N_days_ago = datetime.now() - timedelta(days=N)
    j = str(date_N_days_ago)
    
    return j

def model(symbol):
    
    datefrom = date_from_gen()
    
    df = df2(symbol, datefrom)
    
    df = create_lags(df)
    
    fit = regressor(df)
    
    return fit 

def orders():
    
    
    acct = client.get_margin_account()
    
    symbol = symbols()
    preds = []
    Direction = []
    
    for i in range(len(symbol)):
        pred = float(model(symbol[i]))
        preds.append(pred)
        if pred >= 0:
            Direction.append(float(1))
        if pred < 0:
            Direction.append(float(-1))
            
    data = [preds, Direction, symbol]
    df = pd.DataFrame(data)
    df = df.T
    df.columns = ['Preds', 'Direction', 'Symbol']
    df['Abs'] = df['Preds']*df['Direction']
    Abs = pd.to_numeric(df['Abs'])
    max_index_1 = Abs.idxmax()
    
    Abs[max_index_1] = 0 
    
    max_index_2 = Abs.idxmax()
    
    Abs[max_index_2] = 0 
    
    max_index_3 = Abs.idxmax()
    
    Abs[max_index_3] = 0 
    
    max_index_4 = Abs.idxmax()
    


        
   
    
    
    
    for i in range(len(acct['userAssets'])):
        if acct['userAssets'][i]['asset'] != 'USDT' and acct['userAssets'][i]['asset'] != 'BCHA':
            
            new_cross = (acct['userAssets'][i]['asset']) + 'USDT'
            info = client.get_symbol_info(symbol=new_cross)
            print(new_cross)
            ticker = client.get_symbol_ticker(symbol=new_cross)
            price = float(ticker['price'])
            minimum = float(info['filters'][2]['minQty'])
            desired_b = float(acct['userAssets'][i]['borrowed'])
            if  desired_b > float(11/price):
                
                stepsize = float(info['filters'][2]['stepSize'])
                val_b = desired_b/minimum
                quantity_bs = minimum*math.floor(val_b)
                while quantity_bs < desired_b:    
                    quantity_bs = quantity_bs + minimum
                
                while quantity_bs > desired_b:
                    quantity_bs = quantity_bs - minimum
            
                client.create_margin_order(symbol=new_cross,  side = SIDE_BUY, type = 'MARKET',  sideEffectType = 'AUTO_REPAY', quantity = quantity_bs )
        
            desired_s = float(acct['userAssets'][i]['free'])
            if desired_s > float(11/price):
            
                stepsize = float(info['filters'][2]['stepSize'])
                val_s = desired_s/minimum
                quantity_bs = (minimum*math.floor(val_s))
                while quantity_bs < desired_s:
                    quantity_bs = quantity_bs + minimum
                
                while quantity_bs > desired_s:
                    quantity_bs = quantity_bs - minimum
            
          
                client.create_margin_order(symbol=new_cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_bs))
         
    
    
    
    global amount 
    
     
    for i in range(len(acct['userAssets'])):
        if acct['userAssets'][i]['asset'] == 'USDT' and float(acct['userAssets'][i]['free']) > 100 and float(acct['userAssets'][i]['free']) < 150:
            amount = float(acct['userAssets'][i]['free'])
        else:
            amount = 200
    
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_1] == 1 and acct['userAssets'][i]['asset'] == symbol[max_index_1]:
            cross = symbol[max_index_1] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = (float(amount-10))/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_b = (minimum*(math.floor(val)))
            while quantity_b < desired:
                quantity_b = quantity_b + minimum
                
            while quantity_b > desired:
                quantity_b = quantity_b - minimum
                
    
            print(cross) 
            print(quantity_b)
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_b)
        
      
            
      
        
         
        
        
   
      
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_1] == -1 and acct['userAssets'][i]['asset'] == symbol[max_index_1]:
            cross = symbol[max_index_1] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = float(amount-10)/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_bs = (minimum*(math.floor(val)))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
                
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
            
        #client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))
            client.create_margin_loan(asset=symbol[max_index_1], amount=quantity_bs)
        
            print(cross)
            print(quantity_bs)
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_bs))
        
    
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_2] == 1 and acct['userAssets'][i]['asset'] == symbol[max_index_2]:
            cross = symbol[max_index_2] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = (float(amount-10))/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_b = (minimum*(math.floor(val)))
            while quantity_b < desired:
                quantity_b = quantity_b + minimum
                
            while quantity_b > desired:
                quantity_b = quantity_b - minimum
                
    
            print(cross) 
            print(quantity_b)
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_b)
        
      
            
      
        
         
        
        
   
      
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_2] == -1 and acct['userAssets'][i]['asset'] == symbol[max_index_2]:
            cross = symbol[max_index_2] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = float(amount-10)/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_bs = (minimum*(math.floor(val)))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
                
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
            
        #client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))
            client.create_margin_loan(asset=symbol[max_index_2], amount=quantity_bs)
        
            print(cross)
            print(quantity_bs)
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_bs))
        
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_3] == 1 and acct['userAssets'][i]['asset'] == symbol[max_index_3]:
            cross = symbol[max_index_3] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = (float(amount-10))/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_b = (minimum*(math.floor(val)))
            while quantity_b < desired:
                quantity_b = quantity_b + minimum
                
            while quantity_b > desired:
                quantity_b = quantity_b - minimum
                
    
            print(cross) 
            print(quantity_b)
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_b)
        
      
            
      
        
         
        
        
   
      
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_3] == -1 and acct['userAssets'][i]['asset'] == symbol[max_index_1]:
            cross = symbol[max_index_3] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = float(amount-10)/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_bs = (minimum*(math.floor(val)))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
                
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
            
        #client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))
            client.create_margin_loan(asset=symbol[max_index_3], amount=quantity_bs)
        
            print(cross)
            print(quantity_bs)
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_bs))
        
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_4] == 1 and acct['userAssets'][i]['asset'] == symbol[max_index_4]:
            cross = symbol[max_index_4] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = (float(amount-10))/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_b = (minimum*(math.floor(val)))
            while quantity_b < desired:
                quantity_b = quantity_b + minimum
                
            while quantity_b > desired:
                quantity_b = quantity_b - minimum
                
    
            print(cross) 
            print(quantity_b)
            client.create_margin_order(symbol=cross, side = SIDE_BUY, type = 'MARKET',  quantity= quantity_b)
        
      
            
      
        
         
        
        
   
      
    for i in range(len(acct['userAssets'])):
        if df['Direction'][max_index_4] == -1 and acct['userAssets'][i]['asset'] == symbol[max_index_4]:
            cross = symbol[max_index_4] + 'USDT'
            ticker = client.get_symbol_ticker(symbol=cross)
            cp = float(ticker['price'])
            desired = float(amount-10)/(float(cp)*4)
            info = client.get_symbol_info(symbol=cross)
            minimum = float(info['filters'][2]['minQty'])
            maximum = float(info['filters'][2]['maxQty'])
            val = desired/minimum
            quantity_bs = (minimum*(math.floor(val)))
            while quantity_bs < desired:
                quantity_bs = quantity_bs + minimum
                
            while quantity_bs > desired:
                quantity_bs = quantity_bs - minimum
            
        #client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_s))
            client.create_margin_loan(asset=symbol[max_index_4], amount=quantity_bs)
        
            print(cross)
            print(quantity_bs)
            client.create_margin_order(symbol=cross,  side = SIDE_SELL, type = 'MARKET',  quantity=(quantity_bs))
        
          
    return 


'''acct = client.get_margin_account()
    
print(acct['userAssets'][3]['free']) 
print(acct['userAssets'][3]['asset'])'''



while '1' == '1':
    
    print(orders())
    
    time.sleep(1683)
