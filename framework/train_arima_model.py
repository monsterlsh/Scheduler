import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')

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

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from statsmodels.tsa.stattools import adfuller
from time import time
def forecast_accuracy(forecast_ori, actual_ori):
    forecast = []
    actual = []
    for idx,ele in enumerate(actual_ori):
        if ele != 0:
            actual.append(ele)
            forecast.append(forecast_ori[idx])
    forecast =np.array(forecast)
    actual = np.array(actual)
    mape = 100*np.mean(np.abs(forecast - actual)/np.abs(actual))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean((forecast - actual)/actual)
    rmse = np.mean((forecast - actual)**2)**0.5    # RMSE
    #rmse_1 = np.sqrt(sum((forecast - actual) ** 2) / actual.size)
    corr = np.corrcoef(forecast, actual)[0, 1]
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)
    return ({'mape': mape,
             'me': me,
             'mae': mae,
             'mpe': mpe,
             'rmse': rmse,
             'corr': corr,
             'minmax': minmax
             })

def param_product():
    ps = range(0, 4)
    d=range(0,2) 
    qs = range(0, 4)
    # Ps = range(0, 2)
    # D=range(0,2) 
    # Qs = range(0, 2)
    # s = range(0,24) # season length is still 24
    # creating list with all the possible combinations of parameters
    parameters = product(ps,d, qs)#, Ps,D, Qs,s)
    return list(parameters)


def optimizeSARIMA(parameters_list,cpu,truelist,presize,d=1, D=1, s=24):
    """Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")
    best_mape = float("inf")
    
    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(cpu, order=(param[0], param[1], param[2])
                                            ,seasonal_order=(0,0,0,0)).fit(disp=-1) 
        except:
            continue
        aic = model.aic
        
        #modiy
        forecast = np.array(model.forecast(presize,alpha=0.01))
        
        fore_metric = forecast_accuracy(forecast,truelist)
        # saving best model, AIC and parameters
        if aic < best_aic and fore_metric['mape'] < best_mape:
            best_model = model
            best_aic = aic
            best_param = param
            best_mape = fore_metric['mape']
            print(best_mape)
        if best_mape < 20:
            return best_param,best_mape
    
    return best_param,best_mape
def optimizeSARIMA_demo2(parameters_list,train,presize,d=1, D=1, s=24):
    """Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """

    results = []
    best_aic = float("inf")
    best_mape = float("inf")
    best_param = None
    win = 200 #10小时
    i = win
    start = time()
    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        
            while i < len(train):
                cpu = train[i-win:i]
                test = train[i:i+presize]
                print(f' i = {i}  ')#data: cpu ={cpu} test={test}
                i = i+1
                try:
                    model=sm.tsa.statespace.SARIMAX(cpu, order=(param[0], param[1], param[2])
                                                ,seasonal_order=(0,0,0,0)).fit(disp=-1) 
                    aic = model.aic
        
        #modiy
                    forecast = np.array(model.forecast(presize,alpha=0.01))
                    
                    fore_metric = forecast_accuracy(forecast,test)
                    # saving best model, AIC and parameters
                    if aic < best_aic and fore_metric['mape'] < best_mape:
                        #best_model = model
                        best_aic = aic
                        best_param = param
                        best_mape = fore_metric['mape']
                        rmse = fore_metric['rmse']

                        print(f'in index = {i}: forecast={forecast} metric param = {param} mape = {best_mape}  rmse = {rmse} ')
                        
                except:
                    break
        
                if best_mape < 20:
                    break
                    #return best_param,best_mape
    end = time()            
    print(f'花费 {end - start}s , finally metric param = {param} mape = {best_mape}  ')
    return best_param,best_mape
def train_demo():
    params = param_product()
    #test_file = '/hdd/jbinin/AlibabaData/target/instanceid_1.csv'
    test_file = '/hdd/jbinin/alibaba2018_data_extraction/data/hole/instanceid_1.csv'
    presize = 10
    df = pd.read_csv(test_file,header=None)
    cpu = df.iloc[:,0]
    win =int(len(cpu)*0.125)
    train = cpu[:win].values
    optimizeSARIMA_demo2(params,train,presize)
    #print(train[0:10])
    #model,mape = optimizeSARIMA(params,train,test,len(test))
    # print(mape)
    # forecast = model.predict(presize)
    # evals = forecast_accuracy(forecast,test)
    # print(evals['mape'])
if __name__ == '__main__':
    train_demo()