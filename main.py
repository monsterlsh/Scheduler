import json
import os
# import sys
# sys.path.append('.')
from data.loader import InstanceConfigLoader
from framework.algorithm import ThresholdFirstFitAlgorithm
#from framework.instance import InstanceConfig
from framework.machine import MachineConfig
from framework.simulation import Simulation
from framework.trigger import ThresholdTrigger

if __name__ == '__main__':
    res_struct_filename = os.path.join(os.getcwd(),'struct.json')
    instance_number = 20
    machine_number = int(instance_number/2+instance_number/4)
    machine_configs = []
    for id in range(machine_number):
        machine = MachineConfig(id,300,20,20)
        machine_configs.append(machine)
    macFile = '/Users/lsh/Documents/ecnuIcloud/Trace/alibaba_2018/intp_dir'
    windowsFile = 'D:\Data\workplace\ecnuicloud\Traces\intp_dir\\'
    linux_file = '/hdd/sxy/Trace_alibaba2018/alibaba_2018/intp_dir'
    instance_configs = InstanceConfigLoader(linux_file,instance_number)
    sim = Simulation(machine_configs, instance_configs, ThresholdTrigger(), ThresholdFirstFitAlgorithm())
    sim.run()

    #print(sim.cluster.structure)
    struct = sim.cluster.structure
    with open(res_struct_filename,'w') as file_job:
        json.dump(struct,file_job)
