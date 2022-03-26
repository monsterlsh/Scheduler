from email import header
import pandas as pd
import os
import csv
import numpy as np
import pandas as pd
from framework.instance import InstanceConfig
from framework.machine import MachineConfig
import random
#import dask.dataframe as dd
def read_iterator(filepath,readNumPrecent=0.125):
    cpulist = {}
    memlist = {}
    files = os.listdir(filepath)
    #n = int(800)
    for idx,file in enumerate(files):
        filename = os.path.join(filepath, file)
        ids = int(filename[filename.rfind('_')+1:filename.rfind('.')])
        df = pd.read_csv(filename)
        #cpus = cpus.get_chunk(n)
        #df = pd.read_csv(f,header=None)
        #df.rename(columns={0:'cpu' ,1:'mem'},inplace=True)
        cpu = df.iloc[:,0].values.squeeze().tolist()
        mem = df.iloc[:,1].values.squeeze().tolist()
        cpulist[ids] = cpu
        memlist[ids] = mem
        
        #print(idx,',',filename,', cpu list: ',len(cpulist))
    return cpulist,memlist
    
def old_version(vm_cpu_request_file,instance_number):
    firlist = os.listdir(vm_cpu_request_file)
    
    #print(firlist)
    vm_cpu_requests =[]
    for i,file in enumerate(firlist):
        if i > instance_number:
            break
        subfiledir = os.path.join(vm_cpu_request_file,file)
        with open(subfiledir) as f:
            #print(f.name)
            cpus = pd.read_csv(f).values.squeeze().tolist()
            vm_cpu_requests.append(cpus)
    return vm_cpu_requests

# TODO 添加初始machine
def InstanceConfigLoader(vm_cpu_request_file,instance_number=None):
    instance_configs = {}
    inc_mac_id_file = '/hdd/lsh/Scheduler/data/container_machine_id.csv'
    vm_mac = {}
    machine_configs = {}
    #读取所有vm的资源
    if instance_number is None :
        vm_cpu_requests , vm_mem_requests = read_iterator(vm_cpu_request_file)
    else:
        vm_cpu_requests = old_version(vm_cpu_request_file,instance_number)
    
    # cpulist = df.iloc[:,0]
    # maclist = df.iloc[:,1]
    mac = {}
    #读取第一时刻vm安置的关系
    df = pd.read_csv(inc_mac_id_file,header = None)
    inc_ids = {}
    old_new={}
    for idx,data in df.iterrows():
        inc_id = idx
        inc_ids[inc_id] = data[0]
        old_new[data[0]] = inc_ids[inc_id]
        mac_id = data[1]
        if mac_id in mac:
            mac[mac_id].append(inc_id)
        else:
            mac[mac_id] = [inc_id]
        #inc_ids.append(inc_id)
        
    mac = {k:v for k,v in sorted(mac.items(),key=lambda x:x[0])}
    mac_new = {}
    idx = 0
    for k,v in mac.items():
        mac_new[idx] = v
        idx = idx+1
    mac = mac_new
    inc_num = set()
    for machine_id,data in mac.items():
        #print("[{}]: {}".format(idx,data))
        #mac_id = data['macid']-1
        #vm_mac[instanceid] = mac_id
        machine = MachineConfig(machine_id,100,100,100)
        machine_configs[machine_id] = machine
        for instanceid in data:
            inc_num.add(instanceid)
            cpu_curve = vm_cpu_requests[inc_ids[instanceid]]
            memory_curve = vm_mem_requests[inc_ids[instanceid]]
            disk_curve = np.zeros_like(cpu_curve)
            instance_config = InstanceConfig(machine_id,instanceid, cpu_curve[0], memory_curve[0], disk_curve, cpu_curve, memory_curve)
            instance_configs[instanceid]=instance_config
        #machine_id = machine_id + 1
    # print(machine_configs.keys())
    # print(instance_configs.keys())
    # print(f'machine len={len(machine_configs)},instance len ={len(instance_configs)}')
    
    return instance_configs,machine_configs,old_new