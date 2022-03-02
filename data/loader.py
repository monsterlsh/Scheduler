import pandas as pd
import os
import csv
import numpy as np
from framework.instance import InstanceConfig
from framework.machine import MachineConfig
import random
def read_iterator(filepath,readNumPrecent=0.125):
    cpulist = []
    memlist = []
    files = os.listdir(filepath)
    n = int(69119*0.125)
    for idx,file in enumerate(files):
        filename = os.path.join(filepath, file)
        with open(filename) as f:
            cpus = pd.read_csv(f,iterator=True,header=None)
            cpus = cpus.get_chunk(n)
            cpu = cpus[0].values.squeeze().tolist()
            mem = cpus[1].values.squeeze().tolist()
            cpulist.append(cpu[:int(len(cpu)*readNumPrecent)])
            memlist.append(mem[:int(len(mem)*readNumPrecent)])
        print(idx,',',filename,',',n)
    return cpulist,memlist
    '''
    with os.scandir(filepath) as entries:
        for entry in entries:
            if entry.is_file():
                filename = os.path.join(filepath, entry.name)
                #print(filename)
                with open(filename) as f:
                    cpus = pd.read_csv(f,header=None)
                    cpu = cpus[0].values.squeeze().tolist()
                    mem = cpus[1].values.squeeze().tolist()
                    cpulist.append(cpu)
                    memlist.append(mem)
    '''
    
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
    
def InstanceConfigLoader(vm_cpu_request_file,instance_number=None):
    instance_configs = []
    # vm_cpu_requests = None
    # vm_mem_requests = None
    if instance_number is None :
        vm_cpu_requests , vm_mem_requests = read_iterator(vm_cpu_request_file)
    else:
        vm_cpu_requests = old_version(vm_cpu_request_file,instance_number)
    for instanceid in range(len(vm_cpu_requests)):
        print('读取 container',instanceid)
        cpu_curve = vm_cpu_requests[instanceid]
        if instance_number is None:
            memory_curve = vm_mem_requests[instanceid]
        else:
            memory_curve = np.zeros_like(cpu_curve)
        disk_curve = np.zeros_like(cpu_curve)
        #暂时都放在0号机器上
        instance_config = InstanceConfig(0,instanceid, cpu_curve[0], 0, disk_curve, cpu_curve, memory_curve)
        instance_configs.append(instance_config)
    
    return instance_configs