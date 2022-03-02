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

# Importing everything from forecasting quality metrics
# from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
# from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")


class Arima():
    def __init__(self,cpu):
        self.cpu = cpu
        self.parameters_list=self.param_product()
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