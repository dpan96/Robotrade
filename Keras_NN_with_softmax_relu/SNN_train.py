# -*- coding: utf-8 -*-
"""
Created on Jun 06 04:18:50 2020

@author: Group-2
"""
import os
import sys
import numpy
import math
import talib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def check_for_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def data_preprocessing(df):
        
    df.reset_index(inplace=True, drop=True)   
    try:
        df = df.drop(['Unnamed: 0'], axis=1)
    except:
        pass
    try:
        df = df.drop(['volume'], axis=1)
    except:
        pass
    
    returns = df['close']
    returns = returns.pct_change()
    df = pd.concat([df,
                returns,
                ], axis=1)

    df.columns = ['datetime','close','open', 'high','low','close_diff',]
    df['close_diff'] = df['close_diff'].shift(-1)
    
    open_prices = df['open']
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    diff_prices = df['close_diff']
    
    supported = ["ROCP", "MACD", "RSI", "UO", "BOLL", "MA", "STOCH", "AO", "ROC", "WILLR"]

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
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma25rocp = talib.ROCP(ma25, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma40rocp = talib.ROCP(ma40, timeperiod=1)
            ma50rocp = talib.ROCP(ma50, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            feature.append(ma5rocp)
            feature.append(ma10rocp)
            feature.append(ma20rocp)
            feature.append(ma25rocp)
            feature.append(ma30rocp)
            feature.append(ma40rocp)
            feature.append(ma50rocp)
            feature.append(ma60rocp)
            feature.append((ma5 - close_prices) / close_prices)
            feature.append((ma10 - close_prices) / close_prices)
            feature.append((ma20 - close_prices) / close_prices)
            feature.append((ma25 - close_prices) / close_prices)
            feature.append((ma30 - close_prices) / close_prices)
            feature.append((ma40 - close_prices) / close_prices)
            feature.append((ma50 - close_prices) / close_prices)
            feature.append((ma60 - close_prices) / close_prices)
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
    
    cols_to_norm = list_is
    df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
    
    return df


def next_day_predictions(df):
# This function is just to be used for  training Y coordinate and performing accuracy check
    def var (row):
       if row['close_diff'] <= -0.0010 :
          return '0'
       if row['close_diff'] > +0.00010 :
          return '2'
       else:
           return '1'
       
    df['NextDayPred'] = df.apply(lambda row: var(row), axis=1)
    df.NextDayPred = df.NextDayPred.astype(int)
    
    df.NextDayPred[:-2] = df.NextDayPred[:-2].astype(int)
    print(df['NextDayPred'].value_counts())
    
    df = df.drop(['close_diff'], axis=1)
    df = df.dropna()
    return df


def train_test_split_func(df):
    df = df[:-1]

    df_X = df.iloc[:, 0:50].values
    df_y = df.iloc[:, -1].values
    
    X = df.iloc[:, 0:50].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = X[:20000],X[20001:],y[:20000],y[20001:]
    
    return df_X, df_y, X, y, X_train, X_test, y_train, y_test
    

def seq_model(X_train, y_train):
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()
    
    #Initializing ANN
    classifier = Sequential()
    # input layer and first hidden layer
    classifier.add(Dense(62,activation='tanh',kernel_initializer='uniform',input_dim=50))
    classifier.add(Dropout(0.1))
    
    classifier.add(Dense(62,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    
    classifier.add(Dense(62,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    
    classifier.add(Dense(62,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    
    classifier.add(Dense(3,activation="softmax", kernel_initializer="uniform"))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


def enc (row_test):
   if (row_test['Short'] > row_test['Stay'] and 
       row_test['Short'] > row_test['Buy']  ):
      return '0'
   else:
       if (row_test['Stay'] > row_test['Short'] and 
           row_test['Stay'] > row_test['Buy']    ):
          return '1'
       else:
           if (row_test['Buy'] > row_test['Short'] and 
               row_test['Buy'] > row_test['Stay'] ):
               return '2'
           else:
               return 'nan'
           

def main(dataset = 'AUD_USD_H4_for_keras.csv', operation = 'train', duration = 'H4'):
    
    try:
        check_for_gpu()
    except:
        pass
    # clean and setup the dataset, return: dataframe
    df = pd.read_csv(dataset)
    df = data_preprocessing(df)
    
    # save features
    df.to_csv("Dataset_features_extracted.csv")
    
    # add next day prediction used for training
    df = next_day_predictions(df)
    
    if operation == 'train':
        
        #setup to train
        df_X, df_y, X, y, X_train, X_test, y_train, y_test = train_test_split_func(df)
        df_y = to_categorical(df_y)
        y_train = to_categorical(y_train)
        
        classifier = seq_model(X_train, y_train)
        classifier.fit(X_train, y_train, batch_size=30, epochs=100)
        if duration == 'H4':
            classifier.save('keras_nn_aud_usd_train_with_fit_transform_h4.h5')
        elif duration == 'H1':
            classifier.save('keras_nn_eur_usd_train_with_fit_transform_h1.h5')

    if operation == 'test':
        if duration == 'H4':
            classifier = load_model('keras_nn_aud_usd_train_with_fit_transform_h4.h5')
        elif duration == 'H1':
            classifier = load_model('keras_nn_eur_usd_train_with_fit_transform_h1.h5')

        df = df[:-1]
    
        df_X = df.iloc[:, 0:50].values
        df_y = df.iloc[:, -1].values
        
        X_test = df.iloc[:, 0:50].values
        y_test = df.iloc[:, -1].values
        
        y_pred = classifier.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ['Short', 'Stay', 'Buy']
        y_pred = y_pred.astype(float)
        
        y_pred['Predictions'] = y_pred.apply(lambda row_test: enc(row_test), axis=1)
        predictions = y_pred['Predictions']
        predictions = predictions.astype(int)
        pred = np.array(predictions)
        print(pd.Series(pred).value_counts())
        

if __name__ == '__main__':
    dataset = 'AUD_USD_H4_for_keras.csv'
    operation = 'train'
    duration = 'H4'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        operation = sys.argv[2]
        duration = sys.argv[3]
    main(dataset, operation)