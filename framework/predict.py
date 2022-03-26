#!/usr/bin/env python
# coding: utf-8



from statsmodels import api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import scipy
from scipy import arange
import scipy.io as sio
from scipy.fftpack import fft,ifft
from scipy.stats import pearsonr
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.pylab as mpl

import math
import time
import seaborn
# Load packages
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import pmdarima as pm
# Importing everything from forecasting quality metrics
# from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
# from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")


# 判断周期性
def isPeriodic(tempNorm,acfPara,bMax,bMin):
    periodFlag = 0
    acf = sm.tsa.acf(tempNorm, nlags=len(tempNorm))
    
    peak_ind = signal.argrelextrema(acf,np.greater)[0]
    fwbest = acf[signal.argrelextrema(acf, np.greater)]

    index = -1
    ran = 0
    fwbestlen = len(fwbest)
    if fwbestlen == 0:
        periodFlag = 0
        return periodFlag
    for i in range(ran, fwbestlen):
        if fwbest[i] > 0:
            j = i
            res = 0
            while fwbest[j] > 0:
                j += 1
                if j > fwbestlen - 1:
                    periodFlag = 1
                    return periodFlag
            index = (i + j - 1) // 2
            break;

    fd = peak_ind[index]
    numlist = []
    Q = len(tempNorm) // fd
    if Q == 1:
        periodFlag = 0
        return periodFlag
    else:
        for i in range(Q):
            numlist.append(tempNorm[i * fd: (i + 1) * fd])
            
        listlen = len(numlist)
        flag = 0
        for i in range(1,listlen):
            a = pearsonr(numlist[i-1], numlist[i])[0]
            b = np.mean(numlist[i-1])/np.mean(numlist[i])
            if a < acfPara or  b > bMax or b < bMin:
                flag += 1
                
        if flag <= listlen // 3:
            periodFlag = 1
            return periodFlag
        else:
            periodFlag = 0
            return periodFlag
        
# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
# 周期性：ARIMA（没有系统做平稳性检验 但周期性检测的时候好像已经确定是平稳性序列了？ 不能确定是一阶差分还是二阶差分，所以使用的是bic进行定阶）
# --------- 参数 ----------
# 预测后三个时间的值，选取最大值（CPU利用率）
def ARIMA(cpuArray, preSize, pmax, qmax):
    cpu = np.array(cpuArray).astype(float) # 数据从int类型转为float
    bic_matrix = [] #bic矩阵
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            for d in range(2):
                try: #存在部分报错，所以用try来跳过报错。
                    tmp.append(SARIMAX(cpu, order=(p,d,q)).fit(disp=False).bic) # ARIMA(p,2,q)模型
                except:
                    tmp.append(None)
            bic_matrix.append(tmp)
        
    bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    #print(bic_matrix) 
    p,q = bic_matrix.stack().astype(float).idxmin() #先用stack展平，然后用idxmin找出最小值位置。
    print('BIC最小的p值和q值为：%s、%s\%s' %(p,d,q))

    model = SARIMAX(cpu, order=(p,2,q)).fit(disp=False)
    forecastnum = preSize # 预测窗口，未来5天
    yHat = model.forecast(forecastnum,alpha=0.01) # 提高置信区间为99%
    print(f'the predict value is {yHat} and p d q is ({p,d,q})'  )
    # yHatIndex = 0
    # cpuMax = yHat[yHatIndex]
    # yHyHatIndex = yHatIndex + 1
    # for i in range(1, preSize):
    #     if yHat[yHyHatIndex] > cpuMax:
    #         cpuMax = yHat[yHyHatIndex]
    #     yHyHatIndex = yHatIndex + 1
    return yHat



def auto(history_cpu,w,first=True):
    df = pd.DataFrame(history_cpu)
    df = df.reset_index()['cpu']#.rolling(window=360).mean()#.plot()
    df = pd.DataFrame(df.rolling(window=360).mean(),columns=['cpu'])
    df = df.dropna()
    df = df.reset_index()
    df = pd.DataFrame(df,columns=['cpu'])
    fit = None
    if first:
        fit = pm.auto_arima(df['cpu'], trace=True , suppress_warnings=True, stepwise=False)
    fit(df['cpu'])
    forecast= fit.predict(n_periods=w)
    return forecast

def auto_list(history,w,first=True):
    #if first:
    fit = pm.auto_arima(history, trace=True , suppress_warnings=True, stepwise=False)
    #fit(history)
    forecast= fit.predict(n_periods=w)
    return forecast
# 非周期性：前十个单位的95%值来作为返回（CPU利用率）
# --------- 参数 ----------
# 窗口数 & 预测百分比
def percentilePrediction(cpuArray, edgePercentile, windowSize):
    cpuLen = len(cpuArray)
    begin = cpuLen - windowSize
    cpuArray = cpuArray[begin:cpuLen]
    #print("type",type(edgePercentile))
    percentile = np.percentile(cpuArray, edgePercentile)
    return percentile

def predicter(data, acfPara=0.9, bMax=1.05, bMin=0.95, edgePercentile=0.95, windowSize=10, preSize=3, pmax=5, qmax=5):
    # 判断周期性
    periodFlag = isPeriodic(data,acfPara,bMax,bMin)

    if periodFlag == 0 :
        result = percentilePrediction(data, edgePercentile, windowSize)
    elif periodFlag == 1:
        result = ARIMA(data, preSize, pmax, qmax)
    
    return result

#********************** testing **********************
if __name__ == '__main__':
    acfPara = 0.9
    bMax = 1.05
    bMin = 0.95

    edgePercentile = 0.95
    windowSize = 10
    preSize = 3
    pmax = 5
    qmax = 5

    #filepath = [0.05, 0.04000333333333333, 0.049996666666666675, 0.05, 0.05, 0.05, 0.05, 0.040003333333333335, 0.04, 0.04]
    filepath = [0.11999, 0.04002666666666664, 0.03000333333333333, 0.02000333333333333, 0.049990000000000014, 0.05, 0.04000333333333333, 0.08998333333333333, 0.010026666666666673, 0.059983333333333326]
    filepath = [2674.8060304978917, 3371.1788109723193, 2657.161969121835, 2814.5583226655367, 3290.855749923403, 3103.622791045206, 3403.2011487950185, 2841.438925235243, 2995.312700153925, 3256.4042898633224, 2609.8702933486843, 3214.6409110870877, 2952.1736018157644, 3468.7045537306344, 3260.9227206904898, 2645.5024256492215, 3137.857549381811, 3311.3526531674556, 2929.7762119375716, 2846.05991810631, 2606.47822546165, 3174.9770937667918, 3140.910443979614, 2590.6601484185085, 3123.4299821259915, 2714.4060964141136, 3133.9561758319487, 2951.3288157912752, 2860.3114228342765, 2757.4279640677833]
    result = predicter(filepath, acfPara, bMax, bMin, edgePercentile, windowSize, preSize, pmax, qmax)
    #result = isPeriodic(filepath)
    print(result)
