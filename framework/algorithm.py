from abc import ABC, abstractmethod
import functools
import imp
import random
import numpy as np
from framework.cluster import Cluster 
from framework.predict import predicter
#from framework.random_greedy_cpu import SchedulePolicy
import framework.random_greedy_cpumem as sxy
class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass

class ThresholdFirstFitAlgorithm(Algorithm):
    def __call__(self, cluster:Cluster, clock):
        # for inst in instances_to_reschedule:
        #     if inst.machine.to_schedule:
        #         inst.machine.to_schedule = False
        #         cluster.machines_to_schedule.remove(inst.machine)
        #     inst.machine.pop(inst.id)

        candidate_machine = None
        candidate_inst = None
        machines = cluster.machines
        machines_sort = {k:v for k,v in sorted(machines.items(),key= lambda x:x[1].cpu,reverse=True)}
        instances_to_reschedule = cluster.instances_to_reschedule
        ''' for macid,machine in machines_sort.items():
            for inst in instances_to_reschedule:
                if machine.accommodate_w(inst):
                    candidate_machine = machine
                    candidate_inst = inst
                    break'''
                
        
        candidate_machine = next(iter(machines_sort.values()))
        if instances_to_reschedule:
            candidate_inst = list(instances_to_reschedule)[0]
        return candidate_machine, candidate_inst
    
    def ChooseWhcihInstance(self,simulation,cluster,instances_to_reschedule):
        for inst in instances_to_reschedule:
            if inst.machine.to_schedule:
                cluster.machines_to_schedule.remove(inst.machine)
                inst.machine.pop(inst.id)
            if inst.machine.to_schedule and not simulation.trigger.isOverhead(inst.machine) :
                inst.machine.to_schedule = False
                inst.machine = None
class SchdeulerPolicyAlgorithm(Algorithm):
    def __call__(self, cluster:Cluster, env,first,end):
        self.env = env
        self.cluster = cluster
        self.schedule_sort(first,end)
        pass
    def schedule_sort(self,first,end):
        W = 3
        inc_cpuNow = {}
        self.cluster.instances={k:v for k,v in sorted(self.cluster.instances.items(),key=lambda x:x[0])}
        inc_cpuHistory = {inc_id:inc.cpulist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        
        #TODO 前提是inc里的key值是从小到大排序的
        inc_cpu_pre = np.array([inc.cpulist[end:end+W] for inc_id,inc in self.cluster.instances.items()])
        inc_mem_pre = np.array([inc.memlist[end:end+W] for inc_id,inc in self.cluster.instances.items()])

        N = inc_cpu_pre.shape[0]
        M = self.cluster.t_0.shape[1]
        machine_ids = [v.id for k,v in self.cluster.machines.items()]
        forecast_cpu = []
        forecast_mem = []
        #memory
        inc_memoryHistory = {inc_id:inc.memory_curve[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        
        for inc_id,cpuhist in inc_cpuHistory.items():
            # TODO 这里的预测所调的方法是需要改改的
            #next_cpu = predicter(cpuhist)
            #print()
            next_cpu = inc_cpuHistory[inc_id][-1]
            #print(f'type next is {type(next)} and its value is {next}')
            
            forecast_cpu.append(next_cpu)
            try:
                inc_cpuNow[inc_id] = next_cpu[0]
            except:
                inc_cpuNow[inc_id]=next_cpu
            #其他资源
            memoryHist = inc_memoryHistory[inc_id]
            next_mem = predicter(memoryHist)
            forecast_mem.append(next_mem)
            try:
                inc_cpuNow[inc_id] = next_mem[0]
            except:
                inc_cpuNow[inc_id]=next_mem
        #TODO forecast_cpu 转成list 现在ndarry
        self.cluster.update_cpu(forecast_cpu)
        self.cluster.update_mem(forecast_mem)
        #print(f'cpu= {inc_cpu_pre} \n xt_0 = {self.cluster.t_0}  \n N = {N} , M = {M}')
        #inc_cpuNow = {inc_id:inc.cpulist[self.env.now] for inc_id,inc in self.cluster.instances.items()}
        #print(f'inc_cpuNow\'s type is {type(inc_cpuNow)}')
        #inc_cpuNow_sort = sorted(inc_cpuNow.items(),key=lambda x:x[1])
        #print(f'after sorted inc_cpuNow\'s type is {type(inc_cpuNow)}')
        #self.avgSchedule(machine_ids,inc_cpuNow_sort)
        
        
        #TODO or the schedule use
        placement = sxy.SchedulePolicy( Z=10, K=3, W=3, u=0.5, v=0.5, x_t0=self.cluster.t_0, N=N, cpu=inc_cpu_pre,mem=inc_mem_pre, M=M, CPU_MAX=75, a=0.2)
        # 评价指标
        # 执行当前决策后的负载均衡开销和迁移开销
        # 输入：下一时刻的各VM的资源需求量cpu_t1，每次调度得出的放置决策placement，迁移之前的放置情况x_t0，单位迁移开销E
        eval_bal = sxy.CostOfLoadBalance(inc_cpuHistory[inc_id][-1], x_t1=placement)
        eval_mig = sxy.CostOfMigration(self.cluster.x_t0, placement, E=2)
        self.cluster.update_t0(placement)
        print('N ={0} M={1}, eval_bal = {2} eval_mig = {3}'.format(N,M,eval_bal,eval_mig))
        #print(f'the x_t0 is {self.cluster.t_0}')
       
    def avgSchedule(self,machine_ids:list,inc_cpuNow_sort:list):
        for mac_id in machine_ids:
            self.cluster.machines[mac_id].instances.clear()
        for inc in inc_cpuNow_sort:
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            inc_id ,instance = inc
            mac_set = set()
            mac_num = len(self.cluster.machines.items())
            while True:
                machine_id = random.randint(0,mac_num-1)
                mac_set.add(machine_id)
                machine = self.cluster.machines.get(machine_id, None)
                assert machine is not None
                if machine.add_period_inc(inc_id,instance,self.cluster.instances):
                    #print(f'In period {self.env.now}, instance {inc_id} choose machine {machine_id}')
                    break
                if len(mac_set) == mac_num:
                    machine.add_period_inc(inc_id,instance,self.cluster.instances,True)
                    break
    def SchedulePolicy(self,Z, K, W, x_t0, N, cpu, E, M, CPU_MAX, a):
        pass
