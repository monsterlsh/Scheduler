from abc import ABC, abstractmethod

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
        

    def __call__(self, cluster, clock, cpu_threshold=0.75, memory_threshold=0.55, disk_threshold=0.55):
        self.cpu_threshold = cpu_threshold
    
        cluster.machines_to_schedule.clear()
        for ids,machine in cluster.machines.items():
            
            if machine.cpu < 0 or machine.cpu / machine.cpu_capacity <= 1 - cpu_threshold:
                    # or machine.memory / machine.memory_capacity <= (1 - memory_threshold) \
                    # or machine.disk / machine.disk_capacity <= (1 - disk_threshold):
                machine.to_schedule = True
                #print(f'there is trigger , and macid is {machine.id},it\'s cpu left is {machine.cpu} and now is',clock)
                cluster.machines_to_schedule.add(machine)

    def isOverhead(self,machine):
        return  machine.cpu < 0 or machine.cpu / machine.cpu_capacity < 1 - self.cpu_threshold

    def sxy(self,cluster,clock):
        self.clock = clock
        
