import json
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from time import time
def main():
    with open('/hdd/lsh/Scheduler/arima/json/inc_model_cpu.json','r') as f:
        inc_model = json.load(f)
    
    print(len(inc_model))
    path = '/hdd/jbinin/alibaba2018_data_extraction/data/hole/'
    modelpath = '/hdd/lsh/Scheduler/arima/models_cpu/'
    
    names = 'instanceid_'
    ids = 0
    for k,v in inc_model.items():
        if ids > 10 :
            break
        start = time()
        filename = names+str(k)+'.csv'
        modelfile = modelpath+str(v) +'.pkl'
        #print(filename,modelfile)
        instance_path = os.path.join(path,filename)
        df = pd.read_csv(instance_path,header=None)
        lens = 2
        w =3
        cpulist = df[0][:lens]
        #print(k,modelfile)
        model = ARIMAResults.load(modelfile).apply(np.array(cpulist).tolist(),refit=True)
        next_cpu = np.array(model.forecast(w,alphas=0.01))
        actual = df[0][lens:lens + w]
        if (next_cpu < 0.001).any():
            next_cpu = np.zeros(w)
        print(next_cpu.tolist() , actual.values.tolist())
        ids += 1
        # model = ARIMAResults.load(modelfile)#.apply(cpulist,refit=True)
        # next_cpu = np.array(model.forecast(w,alphas=0.01))
        # lens = int(len(df[0])*0.7)
        # actual = df[0][lens:lens + w]
        # print(next_cpu , actual)
        # model = model.append([0.3,0.5,0.9,0.1,0.5])
        # #pqd = model.summary().tables[0].data[1][1]
        end = time()
        print(end-start,'s')
        # print(next_cpu,'consuming:',end-start)
        # next_cpu = np.array(model.forecast(3,alphas=0.01))
        # print('next next',next_cpu)
        #forecast = model.forecast(10,alphas=0.01)
def config_pkl():
    path = '/hdd/jbinin/alibaba2018_data_extraction/data/hole/'
    filename = 'instanceid_841.csv'
    instance_path = os.path.join(path,filename)
    df = pd.read_csv(instance_path,header=None)
    cpulist = df[0][:20]
    print(type(cpulist.values))
if __name__ == '__main__':
    config_pkl()