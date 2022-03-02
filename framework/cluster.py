from http.client import ImproperConnectionState
import imp


import time
import numpy as np
import random
from framework.machine import Machine
from framework.instance import Instance

class Cluster(object):
    def __init__(self):
        self.machines = {}
        self.machines_to_schedule = set()
        self.instances_to_reschedule = None
        self.instances = {} #用在periodSchedule
        self.t_0 =None
        self.cpu = None
        self.mem = None
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
            self.instances[inc.id] = inc
            sets = set()
            sets.update(machine_ids)
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            while True:
                machine_id = random.randint(0,len(self.machines.items())-1)
                machine = self.machines.get(machine_id, None)
                assert machine is not None
                if machine.accommodate_pre(instance_config):
                    machine.add_instance(instance_config)
                    print(f'instance {instance_config.id} choose machine {machine_id}')
                    break
                elif machine_id in sets:
                    sets.remove(machine_id)
                # TODO 没有一个machine适合放置该instance
                if len(sets) == 0:
                    break
        self.update_t0()

    def update_t0(self,x_t1=None):
        if x_t1 is None:
            self.N = len(self.instances)
            self.M = len(self.machines)
            #self.t_0 = [[0 for i in range(M)]for j in range(N)]
            self.t_0 = np.zeros(shape=(self.N,self.M))
            for mac in self.machines.values():
                j = mac.id
                for inc_id in mac.instances.keys():
                    self.t_0[inc_id][j] =1
        else:
            #TODO sxy x_t1
            self.t_0 = x_t1
            for macid in range(self.M):
                self.machines[macid].instances.clear()
                for incid in range(self.N):
                    if x_t1[incid][macid] == 1:
                        self.machines[macid].instances[incid] = self.instances[incid]
            pass

    def update_cpu(self,predict_cpulist:list):
        self.cpu = np.array([v for v in predict_cpulist])

    def update_mem(self,predict_memlist:list):
        self.mem = np.array([v for v in predict_memlist])

    @property
    def structure(self):
        return [ 
        {
            'time': time.asctime( time.localtime(time.time()) )
        },
        
        {
           
            i: {
                    'cpu_capacity': m.cpu_capacity,
                    'memory_capacity': m.memory_capacity,
                    # 'disk_capacity': m.disk_capacity,
                    'cpu': m.cpu,
                    'memory': m.memory,
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