# -*- coding: utf-8 -*-
"""
Created on Jun 05 01:56:49 2020

@author: Group-2
"""

import tensorflow as tf
# This statement checks the GPU on the device and turns the Config ON for the code
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import os
import sys
import tensorflow as tf
import numpy
import talib
import math
import pandas as pd
import pickle 
import numpy as np
import seaborn as sns
from functools import *
from collections import Counter
from operator import itemgetter
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import AUC
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, Activation, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback


def create_labels(df, col_name, window_size=11):
    """
    Data seperation is according to the research paper shared by Dr. Matloob on CNN model
    Label BUY => 1, SELL => 0, HOLD => 2 
    """

    counter = 0
    rows_total = len(df)
    labels = np.zeros(rows_total)
    labels[:] = np.nan
    print("Calculating labels")
    pbar = tqdm(total=rows_total)

    while counter < rows_total:
        if counter >= window_size - 1:
            window_begin = counter - (window_size - 1)
            window_end = counter
            window_middle = (window_begin + window_end) / 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[counter] = 0
            elif min_index == window_middle:
                labels[counter] = 1
            else:
                labels[counter] = 2

        counter = counter + 1
        pbar.update(1)

    pbar.close()
    return labels


def get_weight_samp(y_val):
# This function was created to calculate the weights on class weights, which will be used for imbalanced data.

    y_val = y_val.astype(int)  # compute_class_weight needs int labels
    weights_of_class = compute_class_weight('balanced', np.unique(y_val), y_val)
    
    print("real class weights are {}".format(weights_of_class), np.unique(y_val))
    print("value_counts", np.unique(y_val, return_counts=True))
    weights = y_val.copy().astype(float)
    for i in np.unique(y_val):
        weights[weights == i] = weights_of_class[i] 

    return weights

def reshape_as_image(x, width, height):
    temporary_x = np.zeros((len(x), height, width))
    for i in range(x.shape[0]):
        temporary_x[i] = np.reshape(x[i], (height, width))

    return temporary_x

def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # precision = TP/TP+FP, recall = TP/TP+FN
    rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    def get_precision(i, conf_mat):
        print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
        precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
        recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
        tf.add(i, 1)
        return i, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    def condition(i, conf_mat):
        return tf.less(i, 3)

    i = tf.constant(3)
    i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(f.math.multiply(f1s, weights))
    return weighted_f1

def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall 
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))
    
    
def preprocessing_dataset(dataset):
    df1 = pd.read_csv(dataset)
    open_prices = df1['open']
    close_prices = df1['close']
    high_prices = df1['high']
    low_prices = df1['low']
    print(df1)
    supported = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "UO", "BOLL", "MA", "STOCH", "AO", "ROC", "WILLR"]
    
    list_is = ["ROCP1", "ROCP2", "OROCP", "HROCP", "LROCP", "norm_macd", "norm_signal", "norm_hist", "macdrocp", "singalrocp", 
                "histrocp", "rsi6", "rsi12", "rsi24", "rsi6rocp", "rsi12rocp", "rsi24rocp", "UO", "upperBOLL", "middleBOLL",
                "lowerBOLL", "MA5rocp", "MA10rocp", "MA20rocp", "MA25rocp", "MA30rocp", "MA40rocp", "MA50rocp", "MA60rocp", 
                "MA5", "MA10", "MA20", "MA25", "MA30", "MA40", "MA50", "MA60", "Slow_stochk", "Slow_stochd", "Fast_stochk",
                "Fast_stochd", "Fast_stoch_rsik", "Fast_stoch_rsid", "AO","ROC5", "ROC10", "ROC20", "ROC25", "WILLR"]
    
    feature = []
    
    def extract_by_type(feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None):
    # This function defines the features which are categorized with supporting technical indicators
    
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
    df.insert(0,"close",close_prices)
    df.insert(0,"low",low_prices)
    df.insert(0,"high",high_prices)
    df.insert(0,"open",open_prices)
    
    df = df.dropna()
    df = df.reset_index()
    df.drop(['index'],axis=1)
    df = df.drop(['index'],axis=1)
    
    return df


def prepare_for_training(dataset):
    feature = []
    df = preprocessing_dataset(dataset)
    df['labels'] = create_labels(df, 'close')
    df = df.dropna()
    df = df.reset_index()
    df.drop(['index'],axis=1)
    df = df.drop(['index'],axis=1)
    list_features = list(df.loc[:, 'open':'WILLR'].columns)
    print('Total number of features', len(list_features))
    
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'open':'WILLR'].values, df['labels'].values, train_size=0.8, 
                                                        test_size=0.1, shuffle=False)
    train_split = 0.8
    print('train_split =',train_split)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, train_size=0.8, test_size=1-train_split, 
                                                    shuffle=False)
    
    
    mm_scaler = MinMaxScaler(feature_range=(0, 1)) 
    x_train1 = mm_scaler.fit_transform(x_train)
    
    x_train = mm_scaler.fit_transform(x_train)
    x_cv = mm_scaler.transform(x_cv)
    x_test = mm_scaler.transform(x_test)
    
    x_main = x_train.copy()
    print("Shape of x, y train/cv/test {} {} {} {} {} {}".format(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape))
    
    num_features = 49  # should be a perfect square
    selection_method = 'all'
    topk = 53 if selection_method == 'all' else num_features
    
    
    if selection_method == 'all':
        select_k_best = SelectKBest(f_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_main, y_train)
        
        selected_features_traderbot = itemgetter(*select_k_best.get_support(indices=True))(list_features)
       
    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_cv = select_k_best.transform(x_cv)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_main, y_train)
    
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        print(len(selected_features_mic), selected_features_mic)
        print(select_k_best.get_support(indices=True))
        
    if selection_method == 'all':
        common = list(set(selected_features_traderbot).intersection(selected_features_mic))
        print("common selected features", len(common), common)
        if len(common) < num_features:
            raise Exception('number of common features found {} < {} required features. Increase "topk variable"'.format(len(common), num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:49])
        print(feat_idx)
        
    if selection_method == 'all':
        x_train = x_train[:, feat_idx]
        x_cv = x_cv[:, feat_idx]
        x_test = x_test[:, feat_idx]
    
    print("Shape of x, y train/cv/test {} {} {} {} {} {}".format(x_train.shape, 
                                                                 y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape))
    
    _labels, _counts = np.unique(y_train, return_counts=True)
    print("percentage of class 0 = {}, class 1 = {}".format(_counts[0]/len(y_train) * 100, _counts[1]/len(y_train) * 100))
    
    get_custom_objects().update({"f1_metric": f1_metric, "f1_weighted": f1_weighted})
    weight_samp = get_weight_samp(y_train)
    print("Test weight_samp:", weight_samp)
    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))
    print("y_train",y_train.shape)
    y_cv = one_hot_enc.transform(y_cv.reshape(-1, 1))
    y_test = one_hot_enc.transform(y_test.reshape(-1, 1))
    
    dim = int(np.sqrt(num_features))
    x_train = reshape_as_image(x_train, dim, dim)
    x_cv = reshape_as_image(x_cv, dim, dim)
    x_test = reshape_as_image(x_test, dim, dim)
    # adding a 1-dim for channels (3)
    x_train = np.stack((x_train,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)
    x_cv = np.stack((x_cv,) * 3, axis=-1)
    print("final shape of x, y train/test {} {} {} {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    
    
    fig = plt.figure(figsize=(15, 15))
    columns = rows = 3
    for i in range(1, columns*rows +1):
        index = np.random.randint(len(x_train))
        img = x_train[index]
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.title('image_'+str(index)+'_class_'+str(np.argmax(y_train[index])), fontsize=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.imshow(img)
    plt.show()
    
    return x_train, x_cv, x_test, y_train, y_cv, y_test , weight_samp


def cnn_model_train(x_train,y_train):
    model = Sequential()
    model.add(Conv2D(64, (2,2), input_shape=(x_train[0].shape[0],x_train[0].shape[1], x_train[0].shape[2])))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (2,2)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
            
    model.add(Flatten())
    model.add(Dense(64))
              
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss="binary_crossentropy",
                 optimizer='adam',
                 metrics=['accuracy'])
    model.summary()
    
    return model


def main(dataset = 'AUD_USD_H4.csv', operation = 'train'):
    
    if operation == 'train':
        x_train, x_cv, x_test, y_train, y_cv, y_test, weight = prepare_for_training(dataset)
        classifier = cnn_model_train(x_train,y_train)
        history = classifier.fit(x_train, y_train, batch_size=32, epochs = 100, validation_data=(x_cv, y_cv), sample_weight=weight)
        
        InteractiveShell.ast_node_interactivity = "last"
        
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train_loss', 'accuracy', 'val_loss', 'val_accuracy'], loc='upper left')
        plt.show()
        
        classifier.save('CNN_weighted_model.h5')
    
    if operation == 'test':
        classifier = load_model('CNN_weighted_model.h5')
        feature = []
        df = preprocessing_dataset(dataset)
        df['labels'] = create_labels(df, 'close')
        df = df.dropna()
        df = df.reset_index()
        df.drop(['index'],axis=1)
        df = df.drop(['index'],axis=1)
        list_features = list(df.loc[:, 'open':'WILLR'].columns)
        print('Total number of features', len(list_features))
        x_test = df.loc[:, 'open':'WILLR'].values
        y_test = df['labels'].values
        num_features = 49
        
        mm_scaler = MinMaxScaler(feature_range=(0, 1)) 
        x_test = mm_scaler.fit_transform(x_test)
        x_test = select_k_best.transform(x_test)
        x_test = x_test[:, feat_idx]

        dim = int(np.sqrt(num_features))
        x_test = reshape_as_image(x_test, dim, dim)
        # adding a 1-dim for channels (3)
        x_test = np.stack((x_test,) * 3, axis=-1)
        y_test = one_hot_enc.transform(y_test.reshape(-1, 1))

        pred = classifier.predict(x_test)
        
        pred_classes = np.argmax(pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        check_baseline(pred_classes, y_test_classes)
        conf_mat = confusion_matrix(y_test_classes, pred_classes)
        print(conf_mat)
        labels = [0,1,2]
        
        f1_weighted = f1_score(y_test_classes, pred_classes, labels=None, average='weighted', sample_weight=None)
        print("F1 score (weighted)", f1_weighted)
        print("F1 score (macro)", f1_score(y_test_classes, pred_classes, labels=None, average='macro', sample_weight=None))
        print("F1 score (micro)", f1_score(y_test_classes, pred_classes, labels=None, average='micro', sample_weight=None))
        
        print("Currency_pair_score", cohen_kappa_score(y_test_classes, pred_classes))
        prec = []
        for i, row in enumerate(conf_mat):
            prec.append(np.round(row[i]/np.sum(row), 2))
            print("precision of class {} = {}".format(i, prec[i]))
        print("precision avg", sum(prec)/len(prec))


if __name__ == '__main__':
    dataset = 'AUD_USD_H4.csv'
    operation = 'train'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        operation = sys.argv[2]
    main(dataset, operation)