import json
import sys
sys.path.append('..')
from data.loader import InstanceConfigLoader
from framework.algorithm import ThresholdFirstFitAlgorithm
#from framework.instance import InstanceConfig
from framework.machine import MachineConfig
from framework.simulation import Simulation
from framework.trigger import ThresholdTrigger

if __name__ == '__main__':
    instance_number = 20
    machine_number = int(instance_number/2+instance_number/4)
    machine_configs = []
    for id in range(machine_number):
        machine = MachineConfig(id,300,20,20)
        machine_configs.append(machine)
    # machine_configs = [
    #     MachineConfig(0, 12, 20, 20),
    #     MachineConfig(1, 21, 25, 25)
    # ]  # MachineConfigLoader('./data/machine_resources.a.csv')

    # instance_configs = [
    #     InstanceConfig(0, 0, 3, 5, 5, [3, 6, 5], [5, 5, 5]),
    #     InstanceConfig(1, 0, 3, 5, 5, [3, 2, 1], [5, 5, 5]),
    #     InstanceConfig(2, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
    #     InstanceConfig(3, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
    #     InstanceConfig(4, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5])
    # ]  # InstanceConfigLoader('./data/output_instance_deployed_a.csv')
    macFile = '/Users/lsh/Documents/ecnuIcloud/Trace/alibaba_2018/intp_dir'
    windowsFile = 'D:\Data\workplace\ecnuicloud\Traces\intp_dir\\'
    instance_configs = InstanceConfigLoader(windowsFile,instance_number)
    sim = Simulation(machine_configs, instance_configs, ThresholdTrigger(), ThresholdFirstFitAlgorithm())
    sim.run()

    #print(sim.cluster.structure)
    struct = sim.cluster.structure
    filename = 'D:\\Data\\workplace\\Github\\scheduler\\struct.json'
    with open(filename,'w') as file_job:
        json.dump(struct,file_job)
