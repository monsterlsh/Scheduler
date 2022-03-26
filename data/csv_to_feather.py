from email import header
import pandas as pd
import feather
import numpy as np
import os
import dask.dataframe as dd
#from datetime import datetime as time
import time
def all():
    filepath= '/hdd/jbinin/alibaba2018_data_extraction/data/hole'
    filepath_new='/hdd/lsh/Scheduler/data/hole/'
    files = os.listdir(filepath)
    for idx,file in enumerate(files):
        print(f'{file }')
        filename = os.path.join(filepath, file)
        incname =filename[filename.rfind('/')+1:]
        base_name=os.path.splitext(incname)[0]
        with open(filename) as f:   
            df = pd.read_csv(filename)
            newfilename = filepath_new+base_name +'.feather' 
            feather.write_dataframe(df, newfilename)
        f.close()
            
        
def test():
    filename = '/hdd/lsh/Scheduler/data/container_machine_id.csv'
    filepath='/hdd/lsh/Scheduler/data/hole/'
    base_name=os.path.splitext(filename)[0]
    print(f'{filename }.... base name is{base_name} ')
    with open(filename) as f:
        df = pd.read_csv(f)
        incname = filename[filename.rfind('/')+1:]
        base_name=os.path.splitext(incname)[0]
        newfilename =  filepath+base_name +'.feather'
        feather.write_dataframe(df, newfilename)
    f.close()
    df = pd.read_feather(newfilename)
    print(df.columns[0])

def read_fea():
    file = '/hdd/jbinin/AlibabaData/target/instanceid_18969.csv'
    file = '/hdd/lsh/Scheduler/data/container_machine_id.csv'
    start = time.time()
    df = pd.read_csv(file,header=None)
    cpulist = df.iloc[:,0]
    memlist = df.iloc[:,1]
    #print(memlist)
    cpulist = list(cpulist.items())
    cpuidx = [v for v in range(len(cpulist))]
    mac = {}
    for k,v in memlist.items():
        if v in mac:
            mac[v].append(k)
        else:
            mac[v] = []
            mac[v].append(k)
    maca = {k:v for k,v in sorted(mac.items(),key=lambda x:x[0])}
    maca_new = {}
    i = 0
    for k,v in maca.items():
        maca_new[i] =v
        i=i+1
    for j in range(len(maca)):
        if j not in maca_new:
            print(j)
            
    end = time.time()
    print((end - start))


def read_iterator(filepath='/hdd/lsh/Scheduler/data/hole',readNumPrecent=0.125):
    cpulist = {}
    memlist = {}
    files = os.listdir(filepath)
    n = int(69119*0.125)
    for idx,file in enumerate(files):
        filename = os.path.join(filepath, file)
        
        ids = int(filename[filename.rfind('_')+1:filename.rfind('.')])
        print(filename)
        
        df = pd.read_feather(filename)
        #cpus = cpus.get_chunk(n)
        #df = pd.read_csv(f,header=None)
        #df.rename(columns={0:'cpu' ,1:'mem'},inplace=True)
        cpu = df.iloc[:n,0:1].values.squeeze().tolist()
        mem = df.iloc[:n,1:2].values.squeeze().tolist()
        cpulist[ids] = cpu
        memlist[ids] = mem
        
        print(idx,',',filename,', cpu list: ',len(cpulist))
    return cpulist,memlist
    
if __name__ == '__main__':
    read_fea()