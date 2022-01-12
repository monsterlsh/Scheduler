from abc import ABC, abstractmethod


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