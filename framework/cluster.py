from http.client import ImproperConnectionState
import imp


import time
import random
from framework.machine import Machine
from framework.instance import Instance

class Cluster(object):
    def __init__(self):
        self.machines = {}
        self.machines_to_schedule = set()
        self.instances_to_reschedule = None
        self.instances = {} #用在periodSchedule

    def configure_machines(self, machine_configs):
        for machine_config in machine_configs:
            machine = Machine(machine_config)
            self.machines[machine.id] = machine
            machine.attach(self)
    '''
    初次给host分配vm
    '''
    def configure_instances(self, instance_configs):
        machine_ids = [v.id for k,v in self.machines.items()]
        print('mac: ',machine_ids)
        for instance_config in instance_configs:
            inc = Instance(instance_config)
            self.instances[inc.id] = inc.cpulist 
            sets = set()
            sets.update(machine_ids)
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            while True:
                machine_id = random.randint(0,len(self.machines.items())-1)
                machine = self.machines.get(machine_id, None)
                assert machine is not None
                if machine.accommodate_w(instance_config):
                    machine.add_instance(instance_config)
                    #print(f'instance {instance_config.id} choose machine {machine_id}')
                    break
                elif machine_id in sets:
                    sets.remove(machine_id)
                # TODO 没有一个machine适合放置该instance
                if len(sets) == 0:
                    break

    @property
    def structure(self):
        return [ 
        {
            'time': time.asctime( time.localtime(time.time()) )
        },
        
        {
           
            i: {
                    'cpu_capacity': m.cpu_capacity,
                    # 'memory_capacity': m.memory_capacity,
                    # 'disk_capacity': m.disk_capacity,
                    'cpu': m.cpu,
                    # 'memory': m.memory,
                    # 'disk': m.disk,
                    'instances': {
                        j: {
                            'cpu': inst.cpu,
                            # 'memory': inst.memory,
                            # 'disk': inst.disk
                        } for j, inst in m.instances.items()
                    }
                }
            for i, m in self.machines.items()
        }]