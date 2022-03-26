from abc import ABC, abstractmethod
import numpy as np
# alibaba2018处理后数据中cpu每个值表示该container该时刻需要的cpu资源占96核（每台机器的cpu容量为96核）的百分比，比如8表示所需核数占机器核数的8%
# mem每个值表示该container该时刻需要的mem资源占100（每台机器的mem容量为归一化后的100）的百分比，比如8表示所需内存占机器内存的8%

class Trigger(ABC):
    @abstractmethod
    def __call__(self, cluster, clock):
        pass


class ThresholdTrigger(Trigger):
    def __init__(self) -> None:
        super().__init__()
        self.cpu_threshold = None
        

    def __call__(self, cluster, clock, cpu_threshold=0.15, memory_threshold=0.65, disk_threshold=0.55):
        self.cpu_threshold = cpu_threshold
        #cluster.update_t0()
        cluster.machines_to_schedule.clear()
        for ids,machine in cluster.machines.items():
            
            #print(f'at{clock} ids{ids}trigger: cpuUse={100-machine.cpu} memUse={100-machine.memory}')#,'lens=',len(machine.instances[2].cpulist),'v.cpu=',machine.instances[2].cpulist[0])
            #剩余量d
            if machine.cpu < 0 or machine.cpu  <= ( 1-cpu_threshold)*machine.cpu_capacity \
                    or machine.memory  <= (1-memory_threshold) * machine.memory_capacity :
                    # or machine.disk / machine.disk_capacity <= (1 - disk_threshold):
                machine.to_schedule = True
                #print(f'there is trigger , and macid is {machine.id},it\'s cpu left is {machine.cpu} and now is',clock)
                cluster.machines_to_schedule.add(machine)
                #return True
            

    def isOverhead(self,machine):
        return  machine.cpu < 0 or machine.cpu / machine.cpu_capacity < 1 - self.cpu_threshold

    def sxy(self,cluster,clock):
        self.clock = clock
        
