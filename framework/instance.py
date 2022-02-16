class InstanceConfig(object):
    def __init__(self, instance_id, cpu, memory, disk, cpu_curve=None, memory_curve=None):
        self.id = instance_id
        #self.machine_id = machine_id
        self.cpu = cpu #cpu序列第一个值
        self.memory = memory
        self.disk = disk
        self.cpu_curve = cpu_curve
        self.memory_curve = memory_curve


class Instance(object):
    def __init__(self, instance_config):
        self.id = instance_config.id
        self.cpu = instance_config.cpu
        self.memory = instance_config.memory
        self.disk = instance_config.disk
        self.cpulist = instance_config.cpu_curve
        self.machine = None

    def attach(self, machine):
        self.machine = machine
