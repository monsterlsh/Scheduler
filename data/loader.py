import pandas as pd
import os
import csv
import numpy as np
from framework.instance import InstanceConfig
from framework.machine import MachineConfig
import random

def InstanceConfigLoader(vm_cpu_request_file, instance_number,machine_number=None):
    instance_configs = []
    firlist = os.listdir(vm_cpu_request_file)
    
    #print(firlist)
    vm_cpu_requests = []
    for i,file in enumerate(firlist):
        if i > instance_number:
            break
        subfiledir = os.path.join(vm_cpu_request_file,file)
        with open(subfiledir) as f:
            #print(f.name)
            cpus = pd.read_csv(f).values.squeeze().tolist()
            vm_cpu_requests.append(cpus)
    for instanceid in range(instance_number):
        cpu_curve = vm_cpu_requests[i]
        memory_curve = np.zeros_like(cpu_curve)
        disk_curve = np.zeros_like(cpu_curve)
        instance_config = InstanceConfig(instanceid, cpu_curve[0], 0, disk_curve, cpu_curve, memory_curve)
        instance_configs.append(instance_config)
    
    return instance_configs