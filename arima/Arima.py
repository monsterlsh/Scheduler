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
from  statsmodels.tsa.arima_model  import  ARIMA
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
# from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
# from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")
from  statsmodels.tsa.stattools  import  adfuller  as  ADF

class Arima():
    def __init__(self,filepath,presize):
        self.ver1_readData(filepath,presize)
        self.cpu = self.df['cpulist']
        self.parameters_list=self.param_product()
    
    '''
    sarima vertion 1
    '''
    def param_product(self):
        ps = range(2, 5)
        d=1 
        qs = range(2, 5)
        Ps = range(0, 2)
        D=1 
        Qs = range(0, 2)
        s = 24 # season length is still 24
        # creating list with all the possible combinations of parameters
        parameters = product(ps, qs, Ps, Qs)
        return list(parameters)
   
    def optimizeSARIMA(self, parameters_list,cpu,d, D, s):
        """Return dataframe with parameters and corresponding AIC
            
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order in ARIMA model
            D - seasonal integration order 
            s - length of season
        """
        
        results = []
        best_aic = float("inf")

        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model=sm.tsa.statespace.SARIMAX(cpu, order=(param[0], param[1], param[2]), 
                                                seasonal_order=(param[3], param[4], param[5], s)).fit(disp=-1) 
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
        
        return result_table
    def find_param(self):
        result_table = self.optimizeSARIMA(self.parameters_list, d=1, D=1, s=24)


    # '''
    # arima version2
    # '''
    def ver1_readData(self,filepath,presize):
        df = pd.read_csv(filepath,header=None,iterator=True)
        df = df.get_chunk(presize)
        # df.rename(columns={0:'cpulist'})
        # df.rename(columns={1:'memlist'})
        df.columns=['cpulist','memlist']
        
        self.df =df
        return df
    def ver1_findD(self):
       # Original Series
        fig, axes = plt.subplots(3, 2, sharex=True)
        axes[0, 0].plot(self.df['cpulist']); axes[0, 0].set_title('Original Series')
        plot_acf(self.df['cpulist'], ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(self.df['cpulist'].diff()); axes[1, 0].set_title('1st Order Differencing')
        plot_acf(self.df['cpulist'].diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(self.df['cpulist'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(self.df['cpulist'].diff().diff().dropna(), ax=axes[2, 1])

        plt.show()
    def ver1_findPQ(self):

        pmax = 5
        qmax = 5
        bic_matrix  =  []  #bic矩阵
        for  p  in  range(pmax+1):
            tmp  =  []
            for  q  in  range(qmax+1):  #存在部分报错，所以用try来跳过报错。
                try:
                    tmp.append(ARIMA(self.df["cpuliat"],order=(p,1,q)).fit().bic) 
                except:
                    tmp.append(None)
            bic_matrix.append(tmp)
        bic_matrix  =  pd.DataFrame(bic_matrix)  #从中可以找出最小值
        p,q  =  bic_matrix.stack().idxmin()  
        # #先用stack展平，然后用idxmin找出最小值位置。
        print(u'BIC最小的p值和q值为:%s、%s'  %(p,q))

        #定阶
        # pmax  =  int(len(df["失业率"])/10)  #一般阶数不超过length/10
        # qmax  =  int(len(df["失业率"])/10)  #一般阶数不超过length/10
def ver1_main():
    filepath = '/hdd/jbinin/alibaba2018_data_extraction/data/hole/instanceid_1.csv'
    ver1 = Arima(filepath,3000)
    ver1.ver1_findD()
    pass

if __name__ == "__main__":
    ver1_main()     

