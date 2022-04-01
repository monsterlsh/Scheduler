from http.client import ImproperConnectionState
import imp
from operator import itemgetter


import time
import numpy as np
import pandas as pd
import os
import json
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
        self.t_0 =None # 某时刻container在机器上的位置
        self.cpu = None
        self.mem = None
        self.mac_ids = None # 【新id：旧id】方便调取model
        self.inc_ids = None
        self.inc_model_cpu = None # [旧id:model]
        self.inc_model_mem = None
        self.model_cpu = {} #存储各个cpu arima 模型
        self.model_mem = {}
        self.inccpu_model = {}
        self.incmem_model = {}
        self.startlen = 0
    def configure_machines(self, machine_configs:dict):
       
        for machine_config in machine_configs.values():
            machine = Machine(machine_config)
            self.machines[machine.id] =machine
            machine.attach(self)
    def add_old_new(self,mac_ids,inc_ids):
        self.mac_ids = mac_ids
        self.inc_ids = inc_ids
    '''
    初次给host分配vm
    '''
    def configure_instances(self, instance_configs:dict):
        for instance_config in instance_configs.values():
            inc = Instance(instance_config)
            self.instances[inc.id] = inc
            self.startlen =int(len(inc.cpulist)*0.7)
            self.len = len(inc.cpulist)
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            
            machine_id = inc.mac_id
            machine = self.machines.get(machine_id, None)
            #print(f'macid= {machine_id} inc_id = {inc.id}')
            assert machine is not None
            
            machine.add_instance_init(inc)

        self.update_t0()
    def product_model(self,model_filename,models):
        files = os.listdir(model_filename)
        for idx,file in enumerate(files):
            pkl = os.path.join(model_filename, file)
            ids = file[:file.rfind('.')]
            model = ARIMAResults.load(pkl )
            models[ids] = model
    def configure_pkl(self):
        cpu_jsonfile = '/hdd/lsh/Scheduler/arima/json/inc_model_cpu.json'
        mem_jsonfile = '/hdd/lsh/Scheduler/arima/json/inc_model_mem.json'
        with open(cpu_jsonfile,'r') as fp:
            json_data = json.load(fp)
        with open(mem_jsonfile,'r') as mp:
            json_mem = json.load(mp)
        self.inc_model_cpu = json_data
        self.inc_model_mem = json_mem
        model_filename = '/hdd/lsh/Scheduler/arima/models_cpu'
        model_filename_mem = '/hdd/lsh/Scheduler/arima/models_mem'
        self.product_model(model_filename,self.model_cpu)
        self.product_model(model_filename_mem,self.model_mem)
        flag = False
        #测试
        if flag:
            start = time.time()
            print(f'Simulatinon refit Start!!!!!!!')
            self.refit()
            end = time.time()
            print(f'Simulatinon refit time: {end-start}s')
        else:
            self.just_test_sxy()
        
        
        print(f'modle load finished!')
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
        print('\t\t update x_to consuming ',e-s)

    def update_cpu(self,predict_cpulist):
        self.cpu = np.array(predict_cpulist)

    def update_mem(self,predict_memlist):
        self.mem = np.array(predict_memlist)
    def update_cpu_mem(self,cpulist=None,memlist=None):
        if cpulist is not None and memlist is not None:
            self.cpu = np.array(cpulist)
            self.mem = np.array(memlist)
        elif len(self.instances[0].cpulist) > 0:
            cpulist = { k:inc.cpulist[0] for k, inc in self.instances.items()}
            memlist  = {k:inc.memlist[0] for k, inc in self.instances.items()}
            self.cpu = np.array([ v for k,v in sorted(cpulist.items(),key=lambda x:x[0]) ])
            self.mem = np.array([ v for k, v in sorted(memlist.items(),key=lambda x:x[0] )])

    def just_test_sxy(self):
        for k,v in self.inc_ids.items():
            old_id = str(v)
            model_name = self.inc_model_cpu[old_id]
            inc =self.instances[k]
            assert type(k) is  int
            
            self.inccpu_model[k] = self.model_cpu[model_name]#.apply(cpulist,refit=True)
        for k,v in self.inc_ids.items():
            old_id = str(v)
            model_name = self.inc_model_mem[old_id]
            inc =self.instances[k]
            assert type(k) is  int
            self.incmem_model[k] = self.model_cpu[model_name]#.apply(cpulist,refit=True)
        
    def refit(self):
        for k,v in self.inc_ids.items():
            old_id = str(v)
            model_name = self.inc_model_cpu[old_id]
            
            path = '/hdd/jbinin/AlibabaData/target/'
            filename = 'instanceid_'+old_id+'.csv'
            instance_path = os.path.join(path,filename)
            df = pd.read_csv(instance_path,header=None)[0][:10].values
            
            inc =self.instances[k]
            #print(k,old_id,model_name)
            cpulist = np.array(inc.cpulist[:10])
            flag = (df==cpulist).all()
            assert flag==True
            #print(f'df==cpulist: {flag}')
            assert type(k) is  int
            try:
                self.inccpu_model[k] = self.model_cpu[model_name].apply(cpulist,refit=True)
            except:
                print('wrong : ','old_id = ',old_id,cpulist,'model=',model_name)
        for k,v in self.inc_ids.items():
            old_id = str(v)
            model_name = self.inc_model_mem[old_id]
            inc =self.instances[k]
            cpulist = np.array(inc.memlist[:10])
            assert type(k) is  int
            self.incmem_model[k] = self.model_cpu[model_name].apply(cpulist,refit=True)
        
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