from abc import ABC, abstractmethod
from framework import cluster
import framework.random_greedy_cpumem_simplify as sxy
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
        self.cluster = None

    def __call__(self, cluster, clock, cpu_threshold=0.20, memory_threshold=0.75, disk_threshold=0.55):
        self.cpu_threshold = cpu_threshold
        self.cluster = cluster
        
        cluster.machines_to_schedule.clear()
        
        flag = True
        self.cluster.update_cpu_mem()
        #pm_desc = self.sand_schedule(clock)
        #print(f'\t\t there is trigger pm_desc num is {len(pm_desc)}')
        for ids,machine in cluster.machines.items():
            
            #print(f'Trigger: in time:{clock},mac_{ids} has capacity of cpu left  {machine.cpu}')
            if machine.cpu < 0 or machine.cpu  <= ( 1-cpu_threshold)*machine.cpu_capacity \
                    or machine.memory  <= (1-memory_threshold) * machine.memory_capacity :
                    # or machine.disk / machine.disk_capacity <= (1 - disk_threshold):
                machine.to_schedule = True
                cluster.machines_to_schedule.add(machine)
                
            
    def sand_schedule(self,clock):
        
        x_t0 = self.cluster.t_0 
        cpu_t0 =self.cluster.cpu
        mem_t0 = self.cluster.mem
        CPU_t0 = sxy.ResourceUsageSimplify(cpu_t0, x_t0)
        MEM_t0 = sxy.ResourceUsageSimplify(mem_t0, x_t0)
        Vol_pm = 10000 / ((100 - CPU_t0) * (100 - MEM_t0)) # 注意每PM/VM的资源需求要<100%
        Vol_vm = 10000 / ((100 - cpu_t0) * (100 - mem_t0)) # 1*N矩阵
        VSR = Vol_vm/ mem_t0 # 1*N矩阵
        # print('mem_t0',mem_t0)
        # print('Vol_vm',Vol_vm)
        # print('VSR',VSR)
        # print('Vol_pm',Vol_pm)
        # 机器按Vol值按序排序，存储机器号
        pm_asc = Vol_pm.argsort()
        pm_desc = pm_asc[::-1]
        return pm_desc
    def isOverhead(self,machine):
        return  machine.cpu < 0 or machine.cpu / machine.cpu_capacity < 1 - self.cpu_threshold

    def sxy(self,cluster,clock):
        self.clock = clock
        
