# -*- coding:utf-8 -*-
# @Time : 2021/8/11 14:37
# @Author: 不猜
# @File : a_data.py
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
import statsmodels.formula.api as smf
import statsmodels.api as sm


def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(pct=True)


def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])


class alpha_factor(object):
    def __init__(self, open, close, high, low, turn, pbMRQ, peTTM, psTTM, pcfNcfTTM, vol=None,
                 windows=list([5, 10, 20, 30, 60])):
        print('Calculating alpha')
        alpha = {}
        time_len, goods_num = close.shape
        print(close.shape, open.shape, high.shape, low.shape)
        alpha['KMID'] = self.KMID(close, open).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KLEN'] = self.KLEN(high, low, open).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KMID2'] = self.KMID2(high, low, open, close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KUP'] = self.KUP(high, open, close).transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KUP2'] = self.KUP2(high, open, close, low).transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KLOW'] = self.KLOW(low, open, close).transpose((1, 0)).reshape(goods_num, time_len, 1)

        alpha['KLOW2'] = self.KLOW2(low, high, open, close).transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KSFT'] = self.KSFT(low, high, open, close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['KSFT2'] = self.KSFT2(low, high, close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)

        for window in windows:
            alpha['ROC_' + str(window)] = self.ROC(close, window).values.transpose((1, 0)).reshape(goods_num, time_len,
                                                                                                   1)
            alpha['MA_' + str(window)] = self.MA(close, window).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
            alpha['STD_' + str(window)] = self.STD(close, window).values.transpose((1, 0)).reshape(goods_num, time_len,
                                                                                                   1)

            alpha['BETA_' + str(window)] = self.BETA(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['RSQR_' + str(window)] = self.RSQR(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['RESI_' + str(window)] = self.RESI(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)

            alpha['MAX_' + str(window)] = self.MAX(close, high, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                         time_len, 1)
            alpha['LOW_' + str(window)] = self.LOW(close, low, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                        time_len, 1)
            alpha['QTLU_' + str(window)] = self.QTLU(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)

            alpha['QTLD_' + str(window)] = self.QTLD(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['RANK_' + str(window)] = self.RANK(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['RSV_' + str(window)] = self.RSV(close, low, high, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                              time_len,
                                                                                                              1)

            alpha['IMAX_' + str(window)] = self.IMAX(high, window).values.transpose((1, 0)).reshape(goods_num, time_len,
                                                                                                    1)
            alpha['IMIN_' + str(window)] = self.IMIN(low, window).values.transpose((1, 0)).reshape(goods_num, time_len,
                                                                                                   1)
            alpha['IMXD_' + str(window)] = self.IMXD(high, low, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                         time_len, 1)

            alpha['CNTP_' + str(window)] = self.CNTP(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['CNTN_' + str(window)] = self.CNTN(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['CNTD_' + str(window)] = self.CNTD(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)

            alpha['SUMP_' + str(window)] = self.SUMP(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['SUMN_' + str(window)] = self.SUMN(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
            alpha['SUMD_' + str(window)] = self.SUMD(close, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)

        alpha['open'] = (open / close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['high'] = (high / close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['low'] = (low / close).values.transpose((1, 0)).reshape(goods_num, time_len, 1)

        if vol is not None:

            for window in windows:
                alpha['VMA_' + str(window)] = self.VMA(vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                     time_len, 1)
                alpha['VSTD_' + str(window)] = self.CNTN(vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                       time_len, 1)
                alpha['WVMA_' + str(window)] = self.WVMA(close, vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                              time_len,
                                                                                                              1)

                alpha['CORR_' + str(window)] = self.CORR(close, vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                              time_len,
                                                                                                              1)
                alpha['CORD_' + str(window)] = self.CORD(close, vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                              time_len,
                                                                                                              1)

                alpha['WSUMP_' + str(window)] = self.WSUMP(vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                         time_len, 1)
                alpha['WSUMN_' + str(window)] = self.WSUMN(vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                         time_len, 1)
                alpha['VSUMD_' + str(window)] = self.VSUMD(vol, window).values.transpose((1, 0)).reshape(goods_num,
                                                                                                         time_len, 1)
        alpha['turn'] = turn.values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['pbMRQ'] = pbMRQ.values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['peTTM'] = peTTM.values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['psTTM'] = psTTM.values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['pcfNcfTTM'] = pcfNcfTTM.values.transpose((1, 0)).reshape(goods_num, time_len, 1)

        alpha['dayrate'] = (close.shift(-1) / open.shift(-1) - 1).values.transpose((1, 0)).reshape(goods_num, time_len,
                                                                                                   1)
        alpha['ovntrate'] = (close.shift(-1) / close - 1).values.transpose((1, 0)).reshape(goods_num, time_len, 1)
        alpha['ovntrate2'] = (close.shift(-2) / close.shift(-1) - 1).values.transpose((1, 0)).reshape(goods_num,
                                                                                                      time_len, 1)
        self.alpha_factor = alpha

    def get_feature(self):

        feature = np.concatenate(tuple(self.alpha_factor.values()), axis=2)

        return feature

    def KMID(self, close, open):  # (收盘价-开盘价)/开盘价
        return (close - open) / open

    def KLEN(self, high, low, open):  # (最高价-最低价)/开盘价
        return (high - low) / open

    def KMID2(self, high, low, open, close):  # (收盘价-开盘价)/(最高价-最低价+1e-12)
        return (close - open) / (high - low + 1e-12)

    def KUP(self, high, open, close):
        high = high.values
        open = open.values
        close = close.values
        datenum, futurenum = open.shape
        open = open.reshape(datenum, futurenum, 1)
        close = close.reshape(datenum, futurenum, 1)
        # np.max(np.concatenate((open, close), axis=2), axis=2)是指去open、close中最大的一个
        return (high - np.max(np.concatenate((open, close), axis=2), axis=2)) / open.reshape(datenum, futurenum)

    def KUP2(self, high, open, close, low):
        high = high.values
        open = open.values
        close = close.values
        low = low.values
        datenum, futurenum = open.shape
        open = open.reshape(datenum, futurenum, 1)
        close = close.reshape(datenum, futurenum, 1)
        return (high - np.max(np.concatenate((open, close), axis=2), axis=2)) / (high - low + 1e-12)

    def KLOW(self, low, open, close):
        low = low.values
        open = open.values
        close = close.values
        datenum, futurenum = open.shape
        open = open.reshape(datenum, futurenum, 1)
        close = close.reshape(datenum, futurenum, 1)
        # np.min(np.concatenate((open, close), axis=2), axis=2)是指去open、close中最小的一个
        return (np.min(np.concatenate((open, close), axis=2), axis=2) - low) / open.reshape(datenum, futurenum)

    def KLOW2(self, low, high, open, close):
        low = low.values
        open = open.values
        close = close.values
        high = high.values
        datenum, futurenum = open.shape
        open = open.reshape(datenum, futurenum, 1)
        close = close.reshape(datenum, futurenum, 1)
        return (np.min(np.concatenate((open, close), axis=2), axis=2) - low) / (high - low + 1e-12)

    def KSFT(self, low, high, open, close):
        return (2 * close - high - low) / open

    def KSFT2(self, low, high, close):
        return (2 * close - high - low) / (high - low + 1e-12)

    def ROC(self, close, window):
        return close.shift(periods=window) / close  # 该函数主要的功能就是使数据框中的数据移动

    # 最少需要有值的观测点的数量，对于int类型，默认与window相等。对于offset类型，默认为1。
    # 　　也开始这样理解：取窗口内非空的。超过这个限制就是带上空值计算
    def MA(self, close, window):
        return close.rolling(window=window, min_periods=window).mean() / close

    def STD(self, close, window):
        return close.rolling(window=window, min_periods=window).std() / close

    # OLS最小二乘法，sm.add_constant()。它会在一个 array 左侧加上一列 1
    # results.params获取回归系数
    def LinearRegression_BETA(self, df):
        y = df.values
        x = np.arange(len(y)).reshape(len(y), 1)
        results = sm.OLS(y, sm.add_constant(x)).fit()
        return results.params[1]

    def LinearRegression_Rsquare(self, df):  # 基金中的决定系数
        y = df.values
        x = np.arange(len(y)).reshape(len(y), 1)
        results = sm.OLS(y, sm.add_constant(x)).fit()
        return results.rsquared

    def LinearRegression_Resi(self, df):
        y = df.values
        x = np.arange(len(y)).reshape(len(y), 1)
        results = sm.OLS(y, sm.add_constant(x)).fit()
        return results.resid[-1]  # 残差

    def BETA(self, close, window):
        return close.rolling(window=window, min_periods=window).apply(lambda x: self.LinearRegression_BETA(x)) / close

    def RSQR(self, close, window):
        return close.rolling(window=window, min_periods=window).apply(lambda x: self.LinearRegression_Rsquare(x))

    def RESI(self, close, window):
        return close.rolling(window=window, min_periods=window).apply(lambda x: self.LinearRegression_Resi(x)) / close

    def MAX(self, close, high, window):
        return high.rolling(window=window, min_periods=window).apply(np.max) / close

    def LOW(self, close, low, window):
        return low.rolling(window=window, min_periods=window).apply(np.min) / close

    def QTLU(self, close, window):
        return close.rolling(window=window, min_periods=window).quantile(0.8) / close  # 线性插值，80%的数

    def QTLD(self, close, window):
        return close.rolling(window=window, min_periods=window).quantile(0.2) / close  # 线性插值，20%的数

    def RANK(self, close, window):
        return close.rolling(window=window, min_periods=window).apply(lambda x: rankdata(x)[-1])

    def RSV(self, close, low, high, window):
        min_low = low.rolling(window=window, min_periods=window).min()
        max_high = high.rolling(window=window, min_periods=window).max()
        return (close - min_low) / (max_high - min_low)

    def IMAX(self, high, window):
        return high.rolling(window=window, min_periods=window).apply(np.argmax) / window

    def IMIN(self, low, window):
        return low.rolling(window=window, min_periods=window).apply(np.argmin) / window

    def IMXD(self, high, low, window):
        return (high.rolling(window=window, min_periods=window).apply(np.argmax) - low.rolling(window=window,
                                                                                               min_periods=window).apply(
            np.argmin)) / window

    #  相关系数
    def CORR(self, close, volume, window):
        return close.rolling(window=window, min_periods=window).corr(np.log(1 + volume))

    def CORD(self, close, volume, window):
        ratio_close = close / close.shift(periods=1)
        shift_vol = volume / volume.shift(periods=1)
        return ratio_close.rolling(window=window, min_periods=window).corr(np.log(1 + shift_vol))

    def CNTP(self, close, window):
        shift_close = close.shift(periods=1)
        arise = close > shift_close
        return arise.rolling(window=window, min_periods=window).mean()

    def CNTN(self, close, window):
        shift_close = close.shift(periods=1)
        decent = close < shift_close
        return decent.rolling(window=window, min_periods=window).mean()

    def CNTD(self, close, window):
        shift_close = close.shift(periods=1)
        arise = close > shift_close
        decent = close < shift_close
        return arise.rolling(window=window, min_periods=window).mean() - decent.rolling(window=window,
                                                                                        min_periods=window).mean()

    def SUMP(self, close, window):
        shift_close = close.shift(periods=1)
        sub_close = close - shift_close
        abs_close = np.abs(sub_close)
        abs_values = abs_close.rolling(window=window, min_periods=window).sum()
        sub_close[sub_close < 0] = 0
        sum_values = sub_close.rolling(window=window, min_periods=window).sum()

        return sum_values / (abs_values + 1e-12)

    def SUMN(self, close, window):
        shift_close = close.shift(periods=1)
        sub_close = close - shift_close
        abs_close = np.abs(sub_close)
        abs_values = abs_close.rolling(window=window, min_periods=window).sum()
        sub_close = -1 * sub_close
        sub_close[sub_close < 0] = 0
        sum_values = sub_close.rolling(window=window, min_periods=window).sum()

        return sum_values / (abs_values + 1e-12)

    def SUMD(self, close, window):

        return self.SUMP(close, window) - self.SUMN(close, window)

    def VMA(self, volume, window):
        return volume.rolling(window=window, min_periods=window).mean() / (volume + 1e-12)

    def VSTD(self, volume, window):
        return volume.rolling(window=window, min_periods=window).std() / (volume + 1e-12)

    def WVMA(self, close, volume, window):
        radio_vol = np.abs(close / close.shift(periods=1) - 1) * volume

        return radio_vol.rolling(window=window, min_periods=window).std() / (
                radio_vol.rolling(window=window, min_periods=window).mean() + 1e-12)

    def WSUMP(self, volume, window):
        shift_volume = volume.shift(periods=1)
        sub_volume = volume - shift_volume
        abs_volume = np.abs(sub_volume)
        abs_values = abs_volume.rolling(window=window, min_periods=window).sum()
        sub_volume[sub_volume < 0] = 0
        sum_values = sub_volume.rolling(window=window, min_periods=window).sum()

        return sum_values / (abs_values + 1e-12)

    def WSUMN(self, volume, window):
        shift_volume = volume.shift(periods=1)
        sub_volume = volume - shift_volume
        abs_volume = np.abs(sub_volume)
        abs_values = abs_volume.rolling(window=window, min_periods=window).sum()
        sub_volume = -1 * sub_volume
        sub_volume[sub_volume < 0] = 0
        sum_values = sub_volume.rolling(window=window, min_periods=window).sum()

        return sum_values / (abs_values + 1e-12)

    def VSUMD(self, volume, window):

        return self.SUMP(volume, window) - self.SUMN(volume, window)


if __name__ == '__main__':
    # drop()——删除dataframe中的指定行列
    open = pd.read_csv(r'.\new_results\open.csv', index_col=0).drop(columns=['date'])
    close = pd.read_csv(r'.\new_results\close.csv', index_col=0).drop(columns=['date'])
    max = pd.read_csv(r'.\new_results\high.csv', index_col=0).drop(columns=['date'])
    min = pd.read_csv(r'.\new_results\low.csv', index_col=0).drop(columns=['date'])
    vol = pd.read_csv(r'.\new_results\volume.csv', index_col=0).drop(columns=['date'])
    turn = pd.read_csv(r'.\new_results\turn.csv', index_col=0).drop(columns=['date'])
    pbMRQ = pd.read_csv(r'.\new_results\pbMRQ.csv', index_col=0).drop(columns=['date'])
    peTTM = pd.read_csv(r'.\new_results\peTTM.csv', index_col=0).drop(columns=['date'])
    psTTM = pd.read_csv(r'.\new_results\psTTM.csv', index_col=0).drop(columns=['date'])
    pcfNcfTTM = pd.read_csv(r'.\new_results\pcfNcfTTM.csv', index_col=0).drop(columns=['date'])
    # print(open.shape)
    alpha_feature = alpha_factor(open, close, max, min, turn, pbMRQ, peTTM, psTTM, pcfNcfTTM, vol).get_feature()
    print(alpha_feature.shape)
    np.save(r'alpha_vol.npy', alpha_feature)
