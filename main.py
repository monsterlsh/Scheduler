import json
import os
import sys
sys.path.append('.')
from data.loader import InstanceConfigLoader
from framework.algorithm import ThresholdFirstFitAlgorithm,SchdeulerPolicyAlgorithm
#from framework.instance import InstanceConfig
from framework.machine import MachineConfig
from framework.simulation import Simulation
from framework.trigger import ThresholdTrigger
import pandas as pd
def simple_test(filepath):
    res_struct_filename = os.path.join(os.getcwd(),'struct.json')
    # 测试用
    instance_number = 20
    machine_number = int(instance_number/2+instance_number/4)
    machine_configs = []
    for id in range(machine_number):
        machine = MachineConfig(id,300,20,20)
        machine_configs.append(machine)
    
    instance_configs = InstanceConfigLoader(filepath,instance_number)
    sim = Simulation(machine_configs, instance_configs, ThresholdTrigger(), ThresholdFirstFitAlgorithm(),SchdeulerPolicyAlgorithm())
    sim.run()

    #print(sim.cluster.structure)
    struct = sim.cluster.structure
    with open(res_struct_filename,'w') as file_job:
        json.dump(struct,file_job)
def test_all(filepath):
    res_struct_filename = os.path.join(os.getcwd(),'struct.json')
    machine_number = 4034
    machine_configs = []
    for id in range(machine_number):
        machine = MachineConfig(id,100,100,100)
        machine_configs.append(machine)
    instance_configs = InstanceConfigLoader(filepath)
    sim = Simulation(machine_configs, instance_configs, ThresholdTrigger(), ThresholdFirstFitAlgorithm(),SchdeulerPolicyAlgorithm())
    sim.run()
    struct = sim.cluster.structure
    with open(res_struct_filename,'w') as file_job:
        json.dump(struct,file_job)
    pass

if __name__ == '__main__':
    macFile = '/Users/lsh/Documents/ecnuIcloud/Trace/alibaba_2018/intp_dir'
    windowsFile = 'D:\Data\workplace\ecnuicloud\Traces\intp_dir\\'
    linux_file = '/hdd/jbinin/alibaba2018_data_extraction/data/hole'
    #test_all(linux_file)
    simple_test(linux_file)
    