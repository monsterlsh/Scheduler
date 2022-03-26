from auto_ts import auto_timeseries
import numpy as np
import pandas as pd
filename = '/hdd/jbinin/AlibabaData/target/instanceid_8215.csv'
data = pd.read_csv(filename,header=None)
data.columns=['cpulist','memlist']
lens = int(len(data)*0.7)
traindata = np.array(data[:lens]['cpulist'])

testdata = np.array(data[lens:lens+10]['cpulist'])
model = auto_timeseries(score_type='rmse', 
time_interval='M', 
non_seasonal_pdq=None, 
seasonality=False,        
seasonal_period=12, 
model_type=['best'], 
verbose=2, 
dask_xgboost_flag=0)
model.fit(traindata )#, ts_column,target)
model.predict(testdata, model='best')