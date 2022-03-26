from http.client import ImproperConnectionState
import imp


import time
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMAResults

from sklearn import cluster
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
        self.model = {}
        self.modelfiles={}
    def configure_machines(self, machine_configs:dict):
       
        for machine_config in machine_configs.values():
            machine = Machine(machine_config)
            self.machines[machine.id] =machine
            machine.attach(self)
        
    '''
    初次给host分配vm
    '''
    def configure_instances(self, instance_configs:dict):
        for instance_config in instance_configs.values():
            inc = Instance(instance_config)
            self.instances[inc.id] = inc
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            
            machine_id = inc.mac_id
            machine = self.machines.get(machine_id, None)
            #print(f'macid= {machine_id} inc_id = {inc.id}')
            assert machine is not None
            
            machine.add_instance_init(inc)

        self.update_t0()
    def configure_pkl(self,filepath):
        files = os.listdir(filepath)
        for idx,file in enumerate(files):
            filename = os.path.join(filepath, file)
            ids = int(file[:file.rfind('.')])
            self.modelfiles[ids] = filename
            model = ARIMAResults.load(filename )
            self.model[ids] = model
            
        pass
    def update_model(self,inc_id,model):
        pass
    def configure_model(self,filename):
        
        df = pd.read_csv(filename)
        lens = df.shape[0]
        for i in  range(lens):
            p = df['p'][i]
            d = df['d'][i]
            q = df['q'][i]
            mape = df['mape'][i]
            instancePath = df['file'][i]
            ids = int(instancePath[instancePath.rfind('_')+1:instancePath.find('.')])
            self.model[ids] = tuple([p,d,q,mape])
        pass
    def update_t0(self,x_t1=None):
        s = time.time()
        if self.t_0 is None and x_t1 is None:
            self.N = len(self.instances)
            self.M = len(self.machines)
            print(f'{self.N}  {self.M}')
            #self.t_0 = [[0 for i in range(M)]for j in range(N)]
            self.t_0 = np.zeros(shape=(self.N,self.M))
            for mac in self.machines.values():
                j = mac.id
                # 由于mac——id 表的contaier数量比实际多
                for inc_id in mac.instances.keys():
                    if inc_id <self.N:
                        self.t_0[inc_id][j] =1
        elif x_t1 is None:
            return 
        else:
            #TODO sxy x_t1
            self.t_0 = x_t1
            for macid in range(self.M):
                self.machines[macid].instances.clear()
                ins = np.where(x_t1[:,macid] == 1)[0]
                for incid in ins:
                    self.machines[macid].instances[incid] = self.instances[incid]
        
        e = time.time()
        print('update consuming ',e-s)

    def update_cpu(self,predict_cpulist):
        self.cpu = np.array(predict_cpulist)

    def update_mem(self,predict_memlist):
        self.mem = np.array(predict_memlist)
    def update_cpu_mem(self,cpulist,memlist):
        self.cpu = np.array(cpulist)
        self.mem = np.array(memlist)
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