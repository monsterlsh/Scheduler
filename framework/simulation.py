from pydoc import ispackage
import simpy
import numpy as np
import sys
sys.path.append('..')
from framework.cluster import Cluster
from framework.monitor import Monitor
from framework.scheduler import Scheduler


class Simulation(object):
    def __init__(self, machine_configs, instance_configs, trigger, algorithm,schedulePolicy=None,modelfilename=None):
        self.env = simpy.Environment()
        self.trigger = trigger
        self.cluster = Cluster()
        self.cluster.configure_machines(machine_configs)
        self.cluster.configure_instances(instance_configs)
        # for k,v in self.cluster.machines.items():
        #     ins = np.array([x.cpu for x in v.instances.values()])
        #     print('machine_',k,'sum is',np.sum(ins),ins)
        self.cluster.configure_pkl(modelfilename)
        self.instance_cpu_curves = {
            self.cluster.machines[instance_config.machine_id].instances[instance_config.id]:
                instance_config.cpu_curve for instance_config in instance_configs.values()
        }
        self.instance_curves = {k:[v.cpulist,v.memlist]for k,v in self.cluster.instances.items()}
        #self.instance_memory_curves = {k:v.memlist for k,v in self.cluster.instances.items()}
       # print('all vms of instance_cpu_curves coded in simulation : ',[x.id for x in self.instance_cpu_curves.keys()])
        self.instance_memory_curves = {
            self.cluster.machines[instance_config.machine_id].instances[instance_config.id]:
                instance_config.memory_curve for instance_config in instance_configs.values()
        }
        # for instance, instance_cpu_curve in self.instance_cpu_curves.items():
        #     self.instance_cpu_curves[instance] = instance_cpu_curve.tolist()
        
        # for instance, instance_m_curve in self.instance_memory_curves.items():
        #     self.instance_memory_curves[instance] = instance_m_curve.tolist()
        
        self.monitor = Monitor(self.env, trigger, algorithm)
        self.scheduler = Scheduler(self.env, algorithm,schedulePolicy)

        self.monitor.attach(self)
        self.scheduler.attach(self)

    def run(self,drl=True,sand=False):
        
        #self.env.process(self.scheduler.run_sand())

        if drl:
            self.env.process(self.monitor.run())
            #self.env.process(self.scheduler.run_drl())
        elif sand:
            self.env.process(self.scheduler.run_sand())
        else:
            self.env.process(self.scheduler.run())
        self.env.run()


    def finished(self,isPeriod = False):
        if isPeriod:
            for vmid,vm in self.cluster.instances.items():
                if self.env.now >= len(vm.cpulist) : #TODO the type of cpulist is int. why? 
                    return True
        
        for instance_cpu_curve in self.instance_curves.values():
            if not instance_cpu_curve[0] and not instance_cpu_curve[1]:
                return True
        
        # for instance_memory_curve in self.instance_memory_curves.items():
        #     if not instance_memory_curve:
        #         return True
        return False
    def finished_drl(self):
        for instance_memory_curve in self.instance_memory_curves.values():
            if not instance_memory_curve:
                return True
        for instance_cpu_curve in self.instance_cpu_curves.values():
            if not instance_cpu_curve:
                return True
        return False
