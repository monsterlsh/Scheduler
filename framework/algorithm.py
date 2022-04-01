from abc import ABC, abstractmethod
import random
from nbformat import write
import numpy as np
from framework.cluster import Cluster 
from arima.predict import arimas
from framework import predict as pe
import framework.random_greedy_cpumem_simplify as sxy
import framework.sandpiper as sand
from time import time 
import csv
from statsmodels.tsa.arima.model import ARIMAResults
class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass


'''
SNANDPIPER algortihm
'''
class ThresholdFirstFitAlgorithm(Algorithm):
    def __call__(self, cluster:Cluster, clock):
        # for inst in instances_to_reschedule:
        #     if inst.machine.to_schedule:
        #         inst.machine.to_schedule = False
        #         cluster.machines_to_schedule.remove(inst.machine)
        #     inst.machine.pop(inst.id)
        cluster.update_t0()
        cluster.update_cpu_mem() #更新clock下 每个container的cpu mem值
        self.CPU_MAX = 15
        flag = self.isAllUnderLoad(cluster.t_0,cluster.cpu,cluster.mem,self.CPU_MAX, MEM_MAX=75)
        if flag == False:
            
            value=self.sand(cluster,clock)
            return value
        else :
            print(' \t no schedule ')
        return 0
    def ResourceUsage(self,cpu_t, x_t):
        x_cput = x_t.copy().T
        CPU_t = np.matmul(x_cput,cpu_t)
        return CPU_t
    def isAllUnderLoad(self,x_t, cpu_t, mem_t, CPU_MAX, MEM_MAX):
        CPU_t = self.ResourceUsage(cpu_t, x_t)
        MEM_t = self.ResourceUsage(mem_t, x_t)
    
        is_cpu = (CPU_t < CPU_MAX).all()
        is_mem = (MEM_t < MEM_MAX).all()
        
        print('\t\t\t simulate trigger in algorithm isload?? : ',is_cpu,is_mem)
        is_all = is_cpu and is_mem # 所以资源都不过载才返回True
        return is_all   
     
    def sand(self,cluster:Cluster,clock):
        cluster.update_t0()
        W =1
        #inc_cpu_pre,inc_mem_pre,inc_cpu_actual,inc_mem_actual = arima(cluster,clock,W)
        #cluster.update_cpu_mem()
        N = cluster.t_0.shape[0]
        M = cluster.t_0.shape[1]
        start = time()
        beforeplace = sxy.CostOfLoadBalanceSimplify(cluster.cpu,cluster.mem ,cluster.t_0,b=0.01)
        
        placement = sand.Sandpiper(cluster.t_0,N,cluster.cpu,cluster.mem,M,self.CPU_MAX, MEM_MAX=75)
        flag = np.array_equal(cluster.t_0, placement, equal_nan=False)
        eval_bal = sxy.CostOfLoadBalanceSimplify(cluster.cpu,cluster.mem , placement,b=0.01)
        eval_mig = sxy.CostOfMigration(cluster.t_0, placement,cluster.mem)
        end = time()
        value = eval_bal+(M-1)*eval_mig
        csvfile = '/hdd/lsh/Scheduler/data/sandpiper_15.csv'
        with open(csvfile,'a') as f:
            writer = csv.writer(f)
            writer.writerow([int(clock),beforeplace,eval_bal])
        # with open('/hdd/lsh/Scheduler/metric_sand.log','a') as f:
        #     f.write(f'eval_bal = {eval_bal}   ,  eval_mig = {eval_mig} \n\t\t consuming {end-start}s, time is {clock} \
        #         \n\t\t sum={value} \n')
        # f.close()
        print(f'\t\t x_t0-placement change ?={flag}change?eval_bal = {eval_bal}   ,  eval_mig = {eval_mig} \n\t\t\t consuming {end-start}s, time is {clock} \
                \n\t\t\t sum={value} \n')
        cluster.update_t0(placement)
        return value
    
    '''
    先进先出
    '''
    def ffa(self,cluster,clock):
        candidate_machine = None
        candidate_inst = None
        machines = cluster.machines
        machines_sort = {k:v for k,v in sorted(machines.items(),key= lambda x:x[1].cpu,reverse=True)}
        instances_to_reschedule = cluster.instances_to_reschedul   
        candidate_machine = next(iter(machines_sort.values()))
        if instances_to_reschedule:
            candidate_inst = list(instances_to_reschedule)[0]
        return candidate_machine, candidate_inst
'''
SXY scheduler policy algorithm
随机采样
'''
class SchdeulerPolicyAlgorithm(Algorithm):
    def __call__(self, cluster:Cluster, env,first,end):
        self.env = env
        self.cluster = cluster
        value = self.schedule_sort(first,end)
        return value
    
    def schedule_sort(self,first,end):
        W = 2
        # # 过去时间应该是从头开始
        # inc_cpuHistory = {inc_id:inc.cpulist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        #  #memory
        # inc_memoryHistory = {inc_id:inc.memlist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        inc_cpu_pre,inc_mem_pre,inc_cpu_actual,inc_mem_actual = arimas(self.cluster,end,W)
        N = inc_cpu_pre.shape[0]
        M = self.cluster.t_0.shape[1]
        #TODO or the schedule use
        start = time()
        
        if inc_cpu_pre.shape[1] <= W:
            placement,cost_min = sxy.SchedulePolicy( 10, 10, W, u=0.8, v=0.2, x_t0=self.cluster.t_0, N=N, cpu=inc_cpu_pre,mem=inc_mem_pre, M=M, CPU_MAX=30, MEM_MAX=75,a=0.004,b=0.01)
            after =  time() 
            # 评价指标
            # 执行当前决策后的负载均衡开销和迁移开销
            # 输入：下一时刻的各VM的资源需求量cpu_t1，每次调度得出的放置决策placement，迁移之前的放置情况x_t0，单位迁移开销E
            eval_bal, eval_mig = 0,0
            t=0
            cpu_t = inc_cpu_actual[:, t] 
            mem_t = inc_mem_actual[:, t]
            eval_bal += sxy.CostOfLoadBalanceSimplify(cpu_t,mem_t , placement,b=0.01)
            eval_mig += sxy.CostOfMigration(self.cluster.t_0, placement,mem_t)
            after =  time() 
            diff = np.where(placement != self.cluster.t_0)
            value = eval_bal+(M-1)*eval_mig
            print(f'\t\t At time {end}, eval_bal={eval_bal},eval_mig = {eval_mig}  \
                  \n\t\t the different array is = {diff} \n\t\t \
                        the time consuming in SXY is {(after-start)}. cost_min ={cost_min}  \n\t\t \
                        mertric  = {value}   \n ')
            #print((placement == self.cluster.t_0).all() )
            self.cluster.update_t0(placement)
            del placement
        return value
            
            
            
    
  