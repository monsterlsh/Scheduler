import this

from prometheus_client import instance_ip_grouping_key
from framework.instance import Instance


class MachineConfig(object):
    def __init__(self, machine_id, cpu_capacity, memory_capacity, disk_capacity):# （cpu=None, memory=None, disk=None):
        self.id = machine_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity

        """self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk"""

        self.to_schedule = False


class Machine(object):
    def __init__(self, machine_config):
        self.id = machine_config.id
        self.cpu_capacity = machine_config.cpu_capacity
        self.memory_capacity = machine_config.memory_capacity
        self.disk_capacity = machine_config.disk_capacity
        self.to_schedule = False
        self.cluster = None
        self.instances = {}

    def attach(self, cluster):
        self.cluster = cluster

    def add_period_inc(self,inc_id,cpu,instances:dict,over=False):
        if cpu < self.cpu or over:
            instance = instances[inc_id]
            instance.attach(self)
            self.instances[instance.id] = instance
            return True
        return False

    def add_instance(self, instance_config):
        assert instance_config.cpu <= self.cpu and instance_config.memory <= self.memory #and instance_config.disk <= self.disk
        instance = Instance(instance_config)
        instance_config.machine_id = self.id
        instance.attach(self)
        self.instances[instance.id] = instance
       
    def accommodate_pre(self,instance,cpu_threshold=0.75):
        remaincpu = self.cpu - instance.cpu
        remainmem = self.memory - instance.memory
        return remaincpu > 0 and remainmem>0 and remaincpu > self.cpu_capacity*(1-cpu_threshold)

    def accommodate_w(self, instance, cpu_threshold=0.75, memory_threshold=0.75, disk_threshold=0.75):
        remain = self.cpu - instance.cpu 
        return  remain > 0 and remain >= self.cpu_capacity*(1-cpu_threshold)

    def pop(self, instance_id):
        instance = self.instances.pop(instance_id)
        """self.cpu += instance.cpu
        self.memory += instance.memory
        self.disk += instance.disk"""
        instance.machine = None
        return instance

    def push(self, instance):
        self.instances[instance.id] = instance
        """self.cpu -= instance.cpu
        self.memory -= instance.memory
        self.disk -= instance.disk"""
        instance.attach(self)

    @property
    def cpu(self):
        occupied = 0
        for instance in self.instances.values():
            occupied += instance.cpu
        return self.cpu_capacity - occupied
    @property
    def isOKAddIns(self,cpu_threshold=0.75):
        cur_cpu = self.cpu
        return cur_cpu > 0 and cur_cpu < self.cpu_capacity*cpu_threshold
    @property
    def memory(self):
        occupied = 0
        for instance in self.instances.values():
            occupied += instance.memory
        
        return self.memory_capacity - occupied

    @property
    def disk(self):
        occupied = 0
        for instance in self.instances.values():
            occupied += instance.disk
        return self.disk_capacity - occupied
