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
import pymysql
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf

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


sl = 0.0010
tp = 0.0040
# Connecting to fxcm. 
token = "a051ed2f89fe5ae6c3c2a0070af4243cb55d7357"
con = fxcmpy.fxcmpy(access_token=token, log_level='error', server='demo', log_file= None)

''' keras.models.load_model loads the existing trained model
'''
     
    
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


def run_all(df, predict_pnl=True, print_explained_trades=True, return_dataframe=False, threshold = 0.0020, classifier=True):
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

    for i in supported:
        extract_by_type(i, open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices)

    df = pd.DataFrame(feature)

    df = df.transpose()

    df.columns = list_is

    df.insert(0,"close_diff",diff_prices)
    df.insert(0,"low",low_prices)
    df.insert(0,"high",high_prices)
    df.insert(0,"open",open_prices)
    df.insert(0,"close",close_prices)
    
    def var (row):
      if row['close_diff'] <= -threshold :
          return '0'
      if row['close_diff'] > +threshold :
          return '2'
      else:
          return '1'
      

    df['NextDayPred'] = df.apply(lambda row: var(row), axis=1)
    df.NextDayPred = df.NextDayPred.astype(int)
    df = df.drop(['close_diff'], axis=1)

    df = df.dropna()
    cols_to_norm = ["ROCP1", "ROCP2", "norm_macd", "norm_signal", "norm_hist", "macdrocp", "singalrocp", "histrocp",
            "rsi6", "rsi12", "rsi24", "rsi6rocp", "rsi12rocp", "rsi24rocp", "UO", "upperBOLL", "middleBOLL",
            "lowerBOLL", "MA5rocp", "MA10rocp", "MA20rocp", "MA25rocp", "MA30rocp", "MA40rocp", "MA50rocp", "MA60rocp", 
            "MA5", "MA10", "MA20", "MA25", "MA30", "MA40", "MA50", "MA60", "Slow_stochk", "Slow_stochd", 
            "Fast_stochk", "Fast_stochd", "Fast_stoch_rsik", "Fast_stoch_rsid", "AO","ROC5", "ROC10", "ROC20",
            "ROC25", "WILLR"]
    df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
    
    # This would exclude the last row if one wants to use this variable to test future prediction
    X_next_day = df.iloc[-1,:]
    X_next_day = np.array(X_next_day)
    X_next_day = X_next_day.reshape((1,51))
    X_next_day = np.delete(X_next_day, 15, 1)

    df = df[:-1]

    df_X = df.iloc[:, 0:50].values
    # df_y = df.iloc[:, -1].values

    X = df.iloc[:, 0:50].values
    y = df.iloc[:, -1].values
    # X_train, X_test, y_train, y_test = X[:19000],X[19001:],y[:19000],y[19001:]
   
    df_X = df.iloc[:, 0:50].values
    df_y = df.iloc[:, -1].values # df_y is actual value
    
    y_pred = classifier.predict(df_X)

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
    predictions = predictions.astype(int)



    df.insert(0,'date',df_date)

    pred_nan = []
    for i in range(60):
        pred_nan.append("nan")
    pred_nan = pd.Series(pred_nan)
    prediction = pred_nan.append(y_pred['Predictions'],  ignore_index=True)

    df = df.reset_index(drop=True)

    buy_trades = []
    sell_trades = []

    for i in range(0, len(df)-1):
        if(predictions[i] == 2):
            buy_trades.append((i+1, df['date'][i+1], df['open'][i+1]))
        elif(predictions[i] == 0):
            sell_trades.append((i+1, df['date'][i+1], df['open'][i+1]))

    if predict_pnl is True:
        print("Prediction of top PnL predictions")
        """Loop over combinations of SL/TP paramaters. Here when we initiate we close if the prediction changed before SL or TP hit"""

        best = []    
        #sl_s = [0.0010, 0.0020, 0.0030, 0.0030, 0.0040, 0.0050]
        #tp_s = [0.0040, 0.0030, 0.0040, 0.0050, 0.0060, 0.0075]
        #for sl, tp in zip(sl_s, tp_s):
            
        buy_trades_pnl = []
        sell_trades_pnl = []

        for buy in buy_trades:
            for i in range(buy[0], len(df)):
                if df['high'][i] >= buy[2] + tp:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],df['high'][i],tp,'t/p hit',i))
                    break
                elif df['low'][i] <= buy[2] - sl:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],df['low'][i],-sl,'s/l hit', i))
                    break
                elif predictions[i] == 0:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],df['close'][i],
                                      (df['close'][i]-buy[2]),'prediction change', i))
                    break

        for sell in sell_trades:
            for i in range(sell[0], len(df)):
                if df['low'][i] <= sell[2] - tp:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],df['low'][i],tp,'t/p hit',i))
                    break
                elif df['high'][i] >= sell[2] + sl:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],df['high'][i],-sl,'s/l hit',i))
                    break
                elif predictions[i] == 2:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],df['close'][i],
                                        (sell [2]-df['close'][i]),'prediction change',i))
                    break
                    
        trades = buy_trades_pnl + sell_trades_pnl
        trades.sort()

        good_trades = [trades[0]]
        x = good_trades[0][8]
        for trade in trades:
            if trade[0] > x:
                good_trades.append(trade)
                x = trade[8]

        PnL = sum(x[6] for x in good_trades)
        best.append((PnL,tp, sl))

        print('Total PnL is:', "{0:.4f}".format(PnL), 'stop-loss:', sl, 'take-profit:', tp,',', len(good_trades), 'trades')
            
        #best.sort(reverse=True)
        #print('\nThe best combination of t/p and s/l is --> take-profit:', best[0][1], 'stop-loss:', best[0][2])

    if print_explained_trades is True:
        print("Explained good trades performed")
        #List all trades, entry prices/dates and exit prices/dates and which action occurred for best combination of s/l and t/p

        #sl = best[0][2]
        #tp = best[0][1]
            
        buy_trades_pnl = []
        sell_trades_pnl = []

        for buy in buy_trades:
            for i in range(buy[0], len(df)):
                if df['high'][i] >= buy[2] + tp:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],buy[2]+tp,tp,'t/p hit',i))
                    break
                elif df['low'][i] <= buy[2] - sl:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],buy[2]-sl,-sl,'s/l hit', i))
                    break
                elif predictions[i] == 0:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],df['close'][i],
                                          (df['close'][i]-buy[2]),'prediction change', i))
                    break

        for sell in sell_trades:
            for i in range(sell[0], len(df)):
                if df['low'][i] <= sell[2] - tp:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],sell[2]-tp,tp,'t/p hit',i))
                    break
                elif df['high'][i] >= sell[2] + sl:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],sell[2]+sl,-sl,'s/l hit',i))
                    break
                elif predictions[i] == 2:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],df['close'][i],
                                            (sell [2]-df['close'][i]),'prediction change',i))
                    break
                        
        trades = buy_trades_pnl + sell_trades_pnl
        trades.sort()

        good_trades = [trades[0]]
        x = good_trades[0][8]
        for trade in trades:
            if trade[0] > x:
                good_trades.append(trade)
                x = trade[8]

        PnL = sum(x[6] for x in good_trades)
        best.append((PnL,tp, sl))

        print('  Entry date   ', '     B/S', '   Entry price', '      Exit date  ', '        Exit price', 
              '   PnL  ', '      Comment','\n')
        for trade in good_trades:
            print("{:20}".format(trade[1]),trade[2],'    ',"{:.4f}".format(trade[3]),'    ',"{:20}".format(trade[4]),'   ',
                  "{:.4f}".format(trade[5]),'   ',"{:.4f}".format(trade[6]),'    ',trade[7])
            
        print('\nTotal PnL is:', "{0:.4f}".format(PnL))

    if return_dataframe is True:
        print("Returning dataframe of good trades to use further")
        #List all trades, entry prices/dates and exit prices/dates and which action occurred for best combination of s/l and t/p

        #sl = best[0][2]
        #tp = best[0][1]
            
        buy_trades_pnl = []
        sell_trades_pnl = []

        for buy in buy_trades:
            for i in range(buy[0], len(df)):
                if df['high'][i] >= buy[2] + tp:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],buy[2]+tp,tp,'t/p hit',i))
                    break
                elif df['low'][i] <= buy[2] - sl:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],buy[2]-sl,-sl,'s/l hit', i))
                    break
                elif predictions[i] == 0:
                    buy_trades_pnl.append((buy[0],buy[1],'buy ',buy[2],df['date'][i],df['close'][i],
                                          (df['close'][i]-buy[2]),'prediction change', i))
                    break

        for sell in sell_trades:
            for i in range(sell[0], len(df)):
                if df['low'][i] <= sell[2] - tp:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],sell[2]-tp,tp,'t/p hit',i))
                    break
                elif df['high'][i] >= sell[2] + sl:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],sell[2]+sl,-sl,'s/l hit',i))
                    break
                elif predictions[i] == 2:
                    sell_trades_pnl.append((sell[0],sell[1],'sell',sell[2],df['date'][i],df['close'][i],
                                            (sell [2]-df['close'][i]),'prediction change',i))
                    break
                        
        trades = buy_trades_pnl + sell_trades_pnl
        trades.sort()
        good_trades = [trades[0]]
        x = good_trades[0][8]
        for trade in trades:
            if trade[0] > x:
                good_trades.append(trade)
                x = trade[8]

        PnL = sum(x[6] for x in good_trades)
        best.append((PnL,tp, sl))

        sno = []
        entry_date = []
        bs = []
        entry_price = []
        exit_date = []
        exit_price = []
        pnl = []
        comment = []

        for index, trade in enumerate(good_trades):
            sno.append(index+1)
            entry_date.append("{:20}".format(trade[1]))
            bs.append(trade[2])
            entry_price.append("{:.4f}".format(trade[3]))
            exit_date.append("{:20}".format(trade[4]))
            exit_price.append("{:.4f}".format(trade[5]))
            pnl.append("{:.4f}".format(trade[6]))
            comment.append(trade[7])
            
        df_backtest = {'Si/No.': sno, 'Entry date': entry_date, 'B/S': bs, 'Entry Price': entry_price, 'Exit date': exit_date, 'Exit price': exit_price, 'PnL': pnl, 'Comment': comment}
        df_backtest = pd.DataFrame(data=df_backtest)
            
        print('\nTotal PnL is:', "{0:.4f}".format(PnL))
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)
        pd.options.display.max_rows
        # print(df_backtest)
        return df_backtest
    
    
def make_new_database(dbname, username, password): #DON'T PUT PASSWORD,ETC HERE; this is a function
    # Create a connection object

    databaseServerIP            = "127.0.0.1"  # IP address of the MySQL database server

    databaseUserName            = username     # User name of the database server

    databaseUserPassword        = password          # Password for the database user



    newDatabaseName             = dbname    # Name of the database that is to be created

    charSet                     = "utf8mb4"     # Character set

    cusrorType                  = pymysql.cursors.DictCursor



    connectionInstance   = pymysql.connect(host=databaseServerIP, user=databaseUserName, password=databaseUserPassword,

                                         charset=charSet,cursorclass=cusrorType, port =3308) #ENTER PORT NUMBER HERE 


    # Create a cursor object

    cursorInsatnce        = connectionInstance.cursor()                                     

    # SQL Statement to create a database

    sqlStatement            = "CREATE DATABASE "+newDatabaseName  

    try:
        # Execute the create database SQL statment through the cursor instance
        cursorInsatnce.execute(sqlStatement)
    except:
        pass

    connectionInstance.close()
    

def create_table(username, password, dbname, currency_pair, duration): #DON'T PUT PASSWORD,ETC HERE; this is a function

    dbServerName    = "127.0.0.1"
    dbUser          = username
    dbPassword      = password         ##Change This 
    dbName          = dbname   ##Change this
    charSet         = "utf8mb4"
    cusrorType      = pymysql.cursors.DictCursor
    connectionObject   = pymysql.connect(host=dbServerName, user=dbUser, password=dbPassword,
                                         db=dbName, charset=charSet,cursorclass=cusrorType, port = 3308) #enter port number here 
    cursorObject        = connectionObject.cursor()

    currency_pair = currency_pair[:3] + currency_pair[4:]
    print(currency_pair)
    sqlQuery            = "CREATE TABLE " + currency_pair + duration +" (entrydate varchar(30), bs varchar(7), entryprice float, exitdate varchar(30), exitprice float, pnl float, comment varchar(100))"

    try:
        cursorObject.execute(sqlQuery)
    except:
        pass
    connectionObject.close()
    print("The Table " + currency_pair + " has been created!")
    
    
def store_in_sql(results,username, password, dbname, currency_pair, duration):#DON'T PUT PASSWORD,ETC HERE; this is a function
    
    dbServerName    = "127.0.0.1"
    dbUser          = username
    dbPassword      = password     # CHANGE THIS
    dbName          = dbname   # CHANGE THIS
    charSet         = "utf8mb4"
    cusrorType      = pymysql.cursors.DictCursor
    connectionObject   = pymysql.connect(host=dbServerName, user=dbUser, password=dbPassword,
                                         db=dbName, charset=charSet,cursorclass=cusrorType, port=3308) #enter port number here 
    cursorObject        = connectionObject.cursor()

    currency_pair = currency_pair[:3] + currency_pair[4:]
    print(currency_pair)
    
    for ind,row in results.iterrows():
        sqlQuery            = "INSERT INTO "  + currency_pair + duration + " (entrydate, bs , entryprice , exitdate , exitprice , pnl, comment) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        inserting_vals = (str(row['Entry date']), str(row['B/S']), float(row['Entry Price']), str(row['Exit date']), float(row['Exit price']), float(row['PnL']), str(row['Comment']))    
        ## Inserting values... 
        cursorObject.execute(sqlQuery, inserting_vals)
        connectionObject.commit()
        
    connectionObject.close()
    print("Data is stored in SQL\n\n\n\n\n")
    

def main(operation = 'run_for', currency_pair = 'all', duration = 'all', past_rows = 200, dbname = 'final2', username = 'root', password = ''):
    
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
                
    '''
       
    if duration == 'all':
        classifier1 = keras.models.load_model('keras_nn_eur_usd_h1_train_with_fit_transform_.h5')
        classifier2 = keras.models.load_model('keras_nn_aud_usd_h4_train_with_fit_transform_.h5')
    else:
        print("Choose duration 'all'")
        classifier1 = False
        classifier2 = False
        
    make_new_database(dbname= dbname, username=username, password= password) 
    duration1 = 'H1'
    duration2 = 'H4'
    past_rows = int(past_rows)

    while True:
        #now = datetime.datetime.now()
        #if now.hour in [10, 14 , 18, 22 ]:
         #   past_rows = int(past_rows)
         #   new_data = make_csv(currency_pair, duration, past_rows)
        #    new_data.to_csv('new_data.csv', index=False)
            for currency_pair in ['AUD/USD', 'EUR/AUD', 'GBP/USD', 'EUR/USD']:
                create_table(dbname = dbname, username=username, password= password, currency_pair=currency_pair, duration=duration1)
                create_table(dbname = dbname, username=username, password= password, currency_pair=currency_pair, duration=duration2)
                
                
                new_data1 = make_csv(currency_pair, duration1, past_rows)
                new_data1.to_csv('new_data1.csv', index=False)
                new_data2 = make_csv(currency_pair, duration2, past_rows)
                new_data2.to_csv('new_data2.csv', index=False)
                
                #new_data1 = pd.read_csv('new_data_h1.csv')
                #new_data2 = pd.read_csv('new_data.csv')
                
                df1 = new_data1
                df2 = new_data2
                
                results_h1 = run_all(df1, predict_pnl = True, print_explained_trades = True, return_dataframe = True, threshold = 0.0020, classifier = classifier1)
                results_h2 = run_all(df2, predict_pnl = True, print_explained_trades = True, return_dataframe = True, threshold = 0.0020, classifier = classifier2)
                
                store_in_sql(results=results_h1, username=username, password=password, dbname= dbname, currency_pair = currency_pair, duration = duration1)
                store_in_sql(results=results_h2, username=username, password=password, dbname= dbname, currency_pair = currency_pair, duration = duration2)
                
            time.sleep(3600)

if __name__ == '__main__':
    operation = 'run_for'
    currency_pair = 'all'
    duration = 'all'
    past_rows = 200
    dbname_str = 'dbname:'
    dbname = 'final2'
    username_str = 'username:'
    username = 'root'
    password_str = 'password:'
    password = ''
    if len(sys.argv) > 1:
        operation = sys.argv[1]
        currency_pair = sys.argv[2]
        duration = sys.argv[3]
        past_rows = sys.argv[4]
        dbname_str = sys.argv[5]
        dbname = sys.argv[6]
        username_str = sys.argv[7]
        username = sys.argv[8]
        password_str = sys.argv[9]
        
    
    main(operation, currency_pair, duration, past_rows, dbname, username, password)
      
    """cd "Documents\Capstone Project\Code_backup\Keras_NN_with_softmax_relu"
        python Keras_NN_live_backtest_strategy.py run_for AUD/USD H1 200 dbname: final1 username: root password:
        python Keras_NN_live_backtest_strategy.py run_for all all 200 dbname: final2 username: root password:
        python Keras_NN_live_backtest_row_by_row_input.py run_for EUR/USD H1 4000
        This code may show an error that you won't be able to run it on command prompt, as that depends on your cuda version and
        computation power. Recommendation would be to run it on jupyter.
    """