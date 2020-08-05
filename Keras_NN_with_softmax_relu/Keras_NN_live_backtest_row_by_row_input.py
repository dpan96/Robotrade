# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:31:53 2020

@author: kohli
"""    
import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: 
    pass

# Important package imports. 
import os
import sys
import math
import time
import talib
import numpy
import pandas
import fxcmpy
import sklearn
import sqlite3
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from collections import deque

import keras
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Connecting to fxcm. 
token = "a051ed2f89fe5ae6c3c2a0070af4243cb55d7357"
con = fxcmpy.fxcmpy(access_token=token, log_level='error', server='demo', log_file= None)

''' keras.models.load_model loads the existing trained model
'''
    
classifier = keras.models.load_model('keras_nn_aud_usd_h4_train_with_fit_transform_.h5')

sl = 0.0010
tp = 0.0040
    
    
##Making the csv
def make_csv(name, dur, num):
    df = con.get_candles(name, period = dur, number = num)
    df.rename(columns={'bidhigh':'high',
                      'bidlow':'low',
                      'bidopen':'open',
                      'bidclose': 'close'}, 
             inplace=True)
    df['dateime'] = df.index
    df = df[['dateime', 'open', 'close', 'low','high']]
    df.reset_index(drop=True, inplace=True)
    new_name = name[:3] + name[4:]
    return df



def run_all(predict_pnl=True, print_explained_trades=True, return_dataframe=False, threshold = 0.0020):
    """ params:
    default predict_pnl = True:
        this variable is True as default and it displays the Profit/Loss pips according to different Stop-Loss and Take_Profit combinations
    default print_explained_trades = True:
        this variable is True as default and it displays the explained good trades the model is making
    default return_dataframe = False:
        this variable is False as default and it returns the explained good trades as a dataframe for for studies and usage
    default threshold = 0.0020:
        this variable is 0.0020 as the H4 model is trained on 0.0020 as a difference between market trends (close_diff)
    """
    #the dataframe is called
    df = pd.read_csv('new_data.csv') 
    df.reset_index(inplace=True, drop=True)
    print(df)

    returns = df['close']
    returns = returns.pct_change()
    df = pd.concat([df,
                    returns,
                    ], axis=1)

    df.columns = ['datetime','open','close', 'low','high','close_diff',]
    df['close_diff'] = df['close_diff'].shift(-1)

    open_prices = df['open']
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    diff_prices = df['close_diff']
    df_date = df['datetime']
    print(df)
    supported = ["ROCP", "MACD", "RSI", "UO", "BOLL", "MA", "STOCH", "AO", "ROC", "WILLR"]

    # extraction of features using supported indicators
    list_is = ["ROCP1", "ROCP2", "norm_macd", "norm_signal", "norm_hist", "macdrocp", "singalrocp", "histrocp",
            "rsi6", "rsi12", "rsi24", "rsi6rocp", "rsi12rocp", "rsi24rocp", "UO", "upperBOLL", "middleBOLL",
            "lowerBOLL", "MA5rocp", "MA10rocp", "MA20rocp", "MA25rocp", "MA30rocp", "MA40rocp", "MA50rocp", "MA60rocp", 
            "MA5", "MA10", "MA20", "MA25", "MA30", "MA40", "MA50", "MA60", "Slow_stochk", "Slow_stochd", 
            "Fast_stochk", "Fast_stochd", "Fast_stoch_rsik", "Fast_stoch_rsid", "AO","ROC5", "ROC10", "ROC20",
            "ROC25", "WILLR"]
    feature = []
    def extract_by_type(feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None):
    
        if feature_type == 'ROCP':
            rocp1 = talib.ROCP(close_prices, timeperiod=1)
            rocp2 = talib.ROCP(close_prices, timeperiod=2)
            feature.append(rocp1)
            feature.append(rocp2)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open_prices, timeperiod=1)
            feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_macd = numpy.nan_to_num(macd) / math.sqrt(numpy.var(numpy.nan_to_num(macd)))
            norm_signal = numpy.nan_to_num(signal) / math.sqrt(numpy.var(numpy.nan_to_num(signal)))
            norm_hist = numpy.nan_to_num(hist) / math.sqrt(numpy.var(numpy.nan_to_num(hist)))
            macdrocp = talib.ROCP(norm_macd + numpy.max(norm_macd) - numpy.min(norm_macd), timeperiod=1)
            signalrocp = talib.ROCP(norm_signal + numpy.max(norm_signal) - numpy.min(norm_signal), timeperiod=1)
            histrocp = talib.ROCP(norm_hist + numpy.max(norm_hist) - numpy.min(norm_hist), timeperiod=1)
            # feature.append(macd / 100.0)
            # feature.append(signal / 100.0)
            # feature.append(hist / 100.0)
            feature.append(norm_macd)
            feature.append(norm_signal)
            feature.append(norm_hist)

            feature.append(macdrocp)
            feature.append(signalrocp)
            feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            feature.append(rsi6 / 100.0 - 0.5)
            feature.append(rsi12 / 100.0 - 0.5)
            feature.append(rsi24 / 100.0 - 0.5)
            # feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            feature.append(rsi6rocp)
            feature.append(rsi12rocp)
            feature.append(rsi24rocp)
        if feature_type == 'UO':
            ult_osc = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            feature.append(ult_osc / 100.0 - 0.5)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            feature.append((upperband - close_prices) / close_prices)
            feature.append((middleband - close_prices) / close_prices)
            feature.append((lowerband - close_prices) / close_prices)         
        if feature_type == 'MA':
            ma5 = talib.MA(close_prices, timeperiod=5)
            ma10 = talib.MA(close_prices, timeperiod=10)
            ma20 = talib.MA(close_prices, timeperiod=20)
            ma25 = talib.MA(close_prices, timeperiod=25)
            ma30 = talib.MA(close_prices, timeperiod=30)
            ma40 = talib.MA(close_prices, timeperiod=40)
            ma50 = talib.MA(close_prices, timeperiod=50)
            ma60 = talib.MA(close_prices, timeperiod=60)
            #ma360 = talib.MA(close_prices, timeperiod=70)
            #ma720 = talib.MA(close_prices, timeperiod=720)
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma25rocp = talib.ROCP(ma25, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma40rocp = talib.ROCP(ma40, timeperiod=1)
            ma50rocp = talib.ROCP(ma50, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            #ma360rocp = talib.ROCP(ma360, timeperiod=1)
            #ma720rocp = talib.ROCP(ma720, timeperiod=1)
            feature.append(ma5rocp)
            feature.append(ma10rocp)
            feature.append(ma20rocp)
            feature.append(ma25rocp)
            feature.append(ma30rocp)
            feature.append(ma40rocp)
            feature.append(ma50rocp)
            feature.append(ma60rocp)
            #feature.append(ma360rocp)
            #feature.append(ma720rocp)
            feature.append((ma5 - close_prices) / close_prices)
            feature.append((ma10 - close_prices) / close_prices)
            feature.append((ma20 - close_prices) / close_prices)
            feature.append((ma25 - close_prices) / close_prices)
            feature.append((ma30 - close_prices) / close_prices)
            feature.append((ma40 - close_prices) / close_prices)
            feature.append((ma50 - close_prices) / close_prices)
            feature.append((ma60 - close_prices) / close_prices)
            #feature.append((ma360 - close_prices) / close_prices)
            #feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'STOCH':
            slow_stoch_k, slow_stoch_d = talib.STOCH(high_prices ,low_prices ,close_prices ,fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            fast_stoch_k, fast_stoch_d = talib.STOCHF(high_prices , low_prices , close_prices , fastk_period=5, fastd_period=3, fastd_matype=0)
            fast_rsi_k, fast_rsi_d = talib.STOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)                      
            feature.append(slow_stoch_k / 100.0 - 0.5)
            feature.append(slow_stoch_d / 100.0 - 0.5)
            feature.append(fast_stoch_k / 100.0 - 0.5)
            feature.append(fast_stoch_d / 100.0 - 0.5)
            feature.append(fast_rsi_k / 100.0 - 0.5)
            feature.append(fast_rsi_d / 100.0 - 0.5)
        if feature_type == 'AO':          
            median_price = (high_prices + low_prices) / 2
            ao = talib.SMA(median_price, 5)-talib.SMA(median_price, 34)          
            feature.append(ao)
        if feature_type == 'ROC':
            roc5 = talib.ROC(close_prices, timeperiod=5)
            roc10 = talib.ROC(close_prices, timeperiod=10)
            roc20 = talib.ROC(close_prices, timeperiod=20)
            roc25 = talib.ROC(close_prices, timeperiod=25)
            feature.append(roc5)
            feature.append(roc10)
            feature.append(roc20)
            feature.append(roc25)
        if feature_type == 'WILLR':
            willr = talib.WILLR(high_prices,low_prices,close_prices, timeperiod=14)
            feature.append(willr / 100.0 - 0.5)

        return feature

    # row by row implementation: referring to Ling's sample code
    buysq = deque([])
    sellsq = deque([])
    
    buy = 0
    sell= 0
    
    #dataframe to store trading records
    col_names =  ['size', 'open_rate', 'open_date','sl','tp','close_rate','close_date','P/L']
    sy1 = pd.DataFrame(columns = col_names)
    sy2 = pd.DataFrame(columns = col_names)
    #setting stop loss and take profit, e.g. 100 pips in this case
    
    
    signal = ""
    count = 0
    count_1 = 0
    
    # sys1 only closes trades when stop loss of take profit reached
    def sys1(sy1, action, price, datetime, sl, tp):
        sy1 = sy1.append({'size': action, 'open_rate': price, 'open_date': datetime, 'sl': price - sl,'tp': price + tp},ignore_index=True)
        return sy1
    
    def check_price(df, price, high, low, datetime, signal):
        #close trades when reach the stoploss or take profit level
        for index, row in df.iterrows():
            if np.isnan(row['close_rate']) and ~np.isnan(row['open_rate']):
                if (price - df['open_rate'][index]) < tp and (price - df['open_rate'][index]) > sl:
                    if signal == 'Buy' or signal == 'Sell':
                        df['close_rate'][index] = price
                        df['close_date'][index] = datetime
                        df['P/L'][index] =  df['close_rate'][index] - df['open_rate'][index]
                        print("profit/loss")
                elif high >= row['tp'] and signal == 'Buy':
                    df['close_rate'][index] = price
                    df['close_date'][index] = datetime
                    df['P/L'][index] = df['close_rate'][index] - df['open_rate'][index]
                    print("T/P")
                elif low <= row['sl'] and signal == 'Buy':
                    df['close_rate'][index] = price
                    df['close_date'][index] = datetime
                    df['P/L'][index] = df['close_rate'][index] - df['open_rate'][index]
                    print("S/L")
                elif low <= row['tp'] and signal == 'Sell':
                    df['close_rate'][index] = price
                    df['close_date'][index] = datetime
                    df['P/L'][index] = df['close_rate'][index] - df['open_rate'][index]
                    print("T/P")
                elif high >= row['tp'] and signal == 'Sell':
                    df['close_rate'][index] = price
                    df['close_date'][index] = datetime
                    df['P/L'][index] = df['close_rate'][index] - df['open_rate'][index]       
                    print("S/L")
    
        return df
    
    open_list = []
    close_list = []
    high_list = []
    low_list = []
    feature = []
    
    
    for index, row in df.iterrows():
        if ~sy1.empty:
            #if it's not empty then check for sl and tp
            sy1 = check_price(sy1, row['open'], row['high'], row['low'], row['datetime'], signal)
        #if buy/sell signal received
        if signal != "":
            if signal == "Buy":
                sy1 = sys1(sy1,2,row['open'],row['datetime'],sl,tp)
            if signal == "Stay":
                # sy1 = sys1(sy1,1,row['open'],row['datetime'],sl,tp)
                signal = ""
            if signal == "Short":
                sy1 = sys1(sy1,0,row['open'],row['datetime'],sl,tp)
                signal = ""
                
        open_list = np.append(open_list,row['open'])
        close_list = np.append(close_list,row['close'])
        high_list = np.append(high_list,row['high'])
        low_list = np.append(low_list,row['low'])
        feature = []
    
        if index > 59:
            for i in supported:
                feature = extract_by_type(i, open_prices=open_list, close_prices=close_list, high_prices=high_list, low_prices=low_list)
            
            df = pd.DataFrame(feature)
            df = df.transpose()
            df.columns = list_is
            df.insert(0,"low",low_list)
            df.insert(0,"high",high_list)
            df.insert(0,"open",open_list)
            df.insert(0,"close",close_list)
            cols_to_norm = ["ROCP1", "ROCP2", "norm_macd", "norm_signal", "norm_hist", "macdrocp", "singalrocp", "histrocp",
                "rsi6", "rsi12", "rsi24", "rsi6rocp", "rsi12rocp", "rsi24rocp", "UO", "upperBOLL", "middleBOLL",
                "lowerBOLL", "MA5rocp", "MA10rocp", "MA20rocp", "MA25rocp", "MA30rocp", "MA40rocp", "MA50rocp", "MA60rocp", 
                "MA5", "MA10", "MA20", "MA25", "MA30", "MA40", "MA50", "MA60", "Slow_stochk", "Slow_stochd", 
                "Fast_stochk", "Fast_stochd", "Fast_stoch_rsik", "Fast_stoch_rsid", "AO","ROC5", "ROC10", "ROC20",
                "ROC25", "WILLR" ]
            df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
            
            df_X = df.iloc[:, 0:50].values
            # open_list = open_list[1:]
            # close_list = close_list[1:]
            # high_list = high_list[1:]
            # low_list = low_list[1:]
            y_pred = classifier.predict(df_X)
            # print(y_pred)
            y_pred = pd.DataFrame(y_pred)
            y_pred.columns = ['Short', 'Stay', 'Buy']
            y_pred = y_pred.astype(float)
    
            # define the predictions
            def enc (row):
              if (row['Short'] > row['Stay'] and 
                  row['Short'] > row['Buy']  ):
                  return '0'
              else:
                  if (row['Stay'] > row['Short'] and 
                      row['Stay'] > row['Buy']    ):
                      return '1'
                  else:
                      if (row['Buy'] > row['Short'] and 
                          row['Buy'] > row['Stay'] ):
                          return '2'
                      else:
                          return 'nan'
                      
    
            y_pred['Predictions'] = y_pred.apply(lambda row: enc(row), axis=1)
            predictions = y_pred['Predictions']
            try:
                predictions = predictions[60:].astype(int)
            except:
                pass
            if predictions[df.index[-1]] == 0:
                signal = "Short"
            if predictions[df.index[-1]] == 1:
                signal = "Stay"
            if predictions[df.index[-1]] == 2:
                signal = "Buy"
            print(signal,index)
            if ~sy1.empty:
                #if it's not empty then check for sl and tp at close price
                sy1 = check_price(sy1, row['close'], row['high'], row['low'], row['datetime'], signal)
    
    total = 0
    count = 0
    profit = 0
    positive = 0
    negative = 0
    for index, row in sy1.iterrows():
        total += 1
        if np.isnan(row['P/L']):
            count += 1
        else:
            #it's divided by 10000 when calculating the sl/tp level, so when calculating the profit, we need to *10000
            profit += row['P/L'] * 10000
            if row['P/L'] > 0:
                positive += 1
            else:
                negative += 1
    print("Total amount of trade made: ", total, ", and", count, "trades not closed")        
    print("There are", positive, "winning trades and", negative, "lossing trades.", "Based on current closed trades, the P/L is : ", profit)
    
    if print_explained_trades is True:
        print(sy1)
        
    if return_dataframe is True:
        return sy1
    

def main(operation = 'run_for', currency_pair = 'AUD/USD', duration = 'H4', past_rows = 1000):
    
    ''' Description and suggestions:
            make_csv('pair_name',x); where pair_name can be any currency pairs with 4 decimal points in format, eg: "AUD/USD"
                                     x is the number of rows you'd like to include as historical data
            new_data.csv is the extracted dataset from the FXCM API
            run_all() includes variables, where:
                default predict_pnl = True:
                    this variable is True as default and it displays the Profit/Loss pips according to different Stop-Loss and Take_Profit combinations
                default print_explained_trades = True:
                    this variable is True as default and it displays the explained good trades the model is making
                default return_dataframe = False:
                    this variable is False as default and it returns the explained good trades as a dataframe for for studies and usage
                default threshold = 0.0020:
                    this variable is 0.0020 as the H4 model is trained on 0.0020 as a difference between market trends (close_diff)
                    we recommend using 0.0015 when we will predict for H1 dataset
    '''
    
    past_rows = int(past_rows)
    new_data = make_csv(currency_pair, duration, past_rows)
    new_data.to_csv('new_data.csv', index=False)
    run_all(predict_pnl = True, print_explained_trades = True, return_dataframe = False, threshold = 0.0020)
    print("Press Ctrl+C to terminate!")


if __name__ == '__main__':
    operation = 'run_for'
    currency_pair = 'AUD/USD'
    duration = 'H4'
    past_rows = 1000
    if len(sys.argv) > 1:
        operation = sys.argv[1]
        currency_pair = sys.argv[2]
        duration = sys.argv[3]
        past_rows = sys.argv[4]
    
    main(operation, currency_pair, duration, past_rows)
    
    """cd "Documents\Capstone Project\Code_backup\Keras_NN_with_softmax_relu"
        python Keras_NN_live_backtest_row_by_row_input.py run_for AUD/USD H4 1000
        python Keras_NN_live_backtest_row_by_row_input.py run_for EUR/USD H1 4000
        This code may show an error that you won't be able to run it on command prompt, as that depends on your cuda version and
        computation power. Recommendation would be to run it on jupyter.
    """