import numpy
import talib
import math
import pandas


class ChartFeature(object):
    def __init__(self, selector):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "UO", "BOLL", "MA", "STOCH", "AO"}
        self.feature = []

    def moving_extract(self, window=30, open_prices=None, close_prices=None, high_prices=None, low_prices=None, with_label=True, flatten=True):
        extract_return = self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices)
        
        df = pandas.DataFrame(extract_return)
        df = df.transpose()
        df.columns = ["ROCP", "OROCP", "HROCP", "LROCP", "norm_macd", "norm_signal", "norm_hist", "macdrocp", "singalrocp", "histrocp",
                    "rsi6", "rsi12", "rsi24", "rsi6rocp", "rsi12rocp", "rsi24rocp", "UO", "upperBOLL", "middleBOLL",
                    "lowerBOLL", "MA5", "MA10", "MA20", "MA30", "MA60", "MA90", "MA120", "MA180",
                    "MA5rocp", "MA10rocp", "MA20rocp", "MA30rocp", "MA60rocp", "MA90rocp", "MA120rocp", "MA180rocp", "Slow_stochk", 
                    "Slow_stochd", "Fast_stochk", "Fast_stochd", "Fast_stoch_rsik", "Fast_stoch_rsid", "AO"]
        df.insert(0,"low",low_prices)
        df.insert(0,"high",high_prices)
        df.insert(0,"close",close_prices)
        df.insert(0,"open",open_prices)
        
        df.to_csv("./Features_output.csv", sep=',',index=False)
        
        feature_arr = numpy.asarray(self.feature)
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" % rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window < feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                # y = cmp(close_prices[p + window], close_prices[p + window - 1]) + 1
                p_change = (close_prices[p + window] - close_prices[p + window - 1]) / close_prices[p + window - 1]
                # use percent of change as label
                y = p_change
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                moving_labels.append(y)
                p += 1

            return numpy.asarray(moving_features), numpy.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                p += 1
            return moving_features

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                print("extracting feature : %s" % feature_type)
                extract_by_type_return = self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices)
                extract_by_type_return
            else:
                print("feature type not supported: %s" % feature_type)
        return self.feature

    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None):
        if feature_type == 'ROCP':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open_prices, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_macd = numpy.nan_to_num(macd) / math.sqrt(numpy.var(numpy.nan_to_num(macd)))
            norm_signal = numpy.nan_to_num(signal) / math.sqrt(numpy.var(numpy.nan_to_num(signal)))
            norm_hist = numpy.nan_to_num(hist) / math.sqrt(numpy.var(numpy.nan_to_num(hist)))
            macdrocp = talib.ROCP(norm_macd + numpy.max(norm_macd) - numpy.min(norm_macd), timeperiod=1)
            signalrocp = talib.ROCP(norm_signal + numpy.max(norm_signal) - numpy.min(norm_signal), timeperiod=1)
            histrocp = talib.ROCP(norm_hist + numpy.max(norm_hist) - numpy.min(norm_hist), timeperiod=1)
            # self.feature.append(macd / 100.0)
            # self.feature.append(signal / 100.0)
            # self.feature.append(hist / 100.0)
            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            # self.feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'UO':
            ult_osc = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            self.feature.append(ult_osc / 100.0 - 0.5)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)         
        if feature_type == 'MA':
            ma5 = talib.MA(close_prices, timeperiod=5)
            ma10 = talib.MA(close_prices, timeperiod=10)
            ma20 = talib.MA(close_prices, timeperiod=20)
            ma30 = talib.MA(close_prices, timeperiod=30)
            ma60 = talib.MA(close_prices, timeperiod=60)
            ma90 = talib.MA(close_prices, timeperiod=90)
            ma120 = talib.MA(close_prices, timeperiod=120)
            ma180 = talib.MA(close_prices, timeperiod=180)
            #ma360 = talib.MA(close_prices, timeperiod=360)
            #ma720 = talib.MA(close_prices, timeperiod=720)
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            #ma360rocp = talib.ROCP(ma360, timeperiod=1)
            #ma720rocp = talib.ROCP(ma720, timeperiod=1)
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            #self.feature.append(ma360rocp)
            #self.feature.append(ma720rocp)
            self.feature.append((ma5 - close_prices) / close_prices)
            self.feature.append((ma10 - close_prices) / close_prices)
            self.feature.append((ma20 - close_prices) / close_prices)
            self.feature.append((ma30 - close_prices) / close_prices)
            self.feature.append((ma60 - close_prices) / close_prices)
            self.feature.append((ma90 - close_prices) / close_prices)
            self.feature.append((ma120 - close_prices) / close_prices)
            self.feature.append((ma180 - close_prices) / close_prices)
            #self.feature.append((ma360 - close_prices) / close_prices)
            #self.feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'STOCH':
            slow_stoch_k, slow_stoch_d = talib.STOCH(high_prices ,low_prices ,close_prices ,fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            fast_stoch_k, fast_stoch_d = talib.STOCHF(high_prices , low_prices , close_prices , fastk_period=5, fastd_period=3, fastd_matype=0)
            fast_rsi_k, fast_rsi_d = talib.STOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)                      
            self.feature.append(slow_stoch_k / 100.0 - 0.5)
            self.feature.append(slow_stoch_d / 100.0 - 0.5)
            self.feature.append(fast_stoch_k / 100.0 - 0.5)
            self.feature.append(fast_stoch_d / 100.0 - 0.5)
            self.feature.append(fast_rsi_k / 100.0 - 0.5)
            self.feature.append(fast_rsi_d / 100.0 - 0.5)
        if feature_type == 'AO':          
            median_price = (high_prices + low_prices) / 2
            ao = talib.SMA(median_price, 5)-talib.SMA(median_price, 34)          
            self.feature.append(ao)
            
        
        return self.feature


def extract_feature(raw_data, selector, window=30, with_label=True, flatten=True):
    chart_feature = ChartFeature(selector)
    sorted_data = sorted(raw_data, key=lambda x:x.datetime)
    opens = []
    closes = []
    highs = []
    lows = []
    for item in sorted_data:
        opens.append(item.open)
        closes.append(item.close)
        highs.append(item.high)
        lows.append(item.low)
    opens = numpy.asarray(opens)
    closes = numpy.asarray(closes)
    highs = numpy.asarray(highs)
    lows = numpy.asarray(lows)
    if with_label:
        moving_features, moving_labels = chart_feature.moving_extract(window=window, open_prices=opens,
                                                                      close_prices=closes,
                                                                      high_prices=highs, low_prices=lows,
                                                                      with_label=with_label, flatten=flatten)
        return moving_features, moving_labels
    else:
        moving_features = chart_feature.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows,
                                                       with_label=with_label, flatten=flatten)
        return moving_features