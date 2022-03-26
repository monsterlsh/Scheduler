from abc import ABC, abstractmethod
import functools
import imp
import random
from nbformat import write
import numpy as np
from pyparsing import alphas
from framework.cluster import Cluster 
from framework import predict as pe
import pmdarima
import gc
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error,mean_absolute_percentage_error
#from framework.random_greedy_cpu import SchedulePolicy
import framework.random_greedy_cpumem_simplify as sxy
import framework.sandpiper as sand
from time import time 
import csv
from statsmodels.tsa.arima.model import ARIMAResults
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
        return self.sand(cluster,clock)
        
        
    def sand(self,cluster:Cluster,clock):
        cluster.update_t0()
        N = cluster.t_0.shape[0]
        M = cluster.t_0.shape[1]
        start = time()
        placement = sand.Sandpiper(cluster.t_0,N,cluster.cpu,cluster.mem,M,CPU_MAX=50, MEM_MAX=75)
        flag = np.array_equal(cluster.t_0, placement, equal_nan=False)
       # print(' \t\t --- flag',flag)
        eval_bal = sxy.CostOfLoadBalanceSimplify(cluster.cpu,cluster.mem , placement,b=0.01)
        eval_mig = sxy.CostOfMigration(cluster.t_0, placement,cluster.mem)
        end = time()
        value = eval_bal+(M-1)*eval_mig
        with open('/hdd/lsh/Scheduler/metric_sand.log','a') as f:
            f.write(f'eval_bal = {eval_bal}   ,  eval_mig = {eval_mig} \n\t\t consuming {end-start}s, time is {clock} \
                \n\t\t sum={value} \n')
        f.close()
        cluster.update_t0(placement)
        return value
    def ffa(self,cluster,clock):
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
        value = self.schedule_sort(first,end)
        return value
    def reduce0(self,forecast,actual):
        index = np.array(np.where(actual==0))
        new_ac = np.delete(actual,index)
        new_fore = np.delete(forecast,index)
        return new_fore,new_ac
    def schedule_sort(self,first,end):
        W = 2
        # # 过去时间应该是从头开始
        # inc_cpuHistory = {inc_id:inc.cpulist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        #  #memory
        # inc_memoryHistory = {inc_id:inc.memlist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        forecast_cpu = {}
        forecast_mem = {}
        startPredict = time()
        iters =0 
        for inc_id,instance in self.cluster.instances.items():
            cpuHist = np.array(instance.cpulist[:end+1])
            memHist = instance.memlist[first:end+1]
            start = time()
            newmape = 0
            flag = False
            # if np.mean(cpuHist[-5:-1]) == 0:
            #     next_cpu = np.zeros(W)
            # else:
            # # TODO 这里的预测所调的方法是需要改改的
            #     if inc_id in self.cluster.model.keys():
            #         flag = True
                    
            #         try:
            #             model =  self.cluster.model[inc_id].apply(cpuHist,refit=True)
            #             next_cpu = model.forecast(W,alphas=0.01)
            #         except:
            #             print(f'container inc_id = {inc_id} 的model不行')
            #             model =  pmdarima.arima.auto_arima(cpuHist)
                        
            #             next_cpu = model.predict(W,alphas=0.01)
            #             print('there is auto')
                        
            #         actual = np.array(instance.cpulist[end+1:end+W+1])
            #         forecast,actual = self.reduce0(next_cpu,actual)
            #         if actual.shape[0] != 0:
            #             newmape = mean_absolute_percentage_error(forecast,actual)
                
                       
            

            # 不预测
            next_cpu = instance.cpulist[end:end+W]
            next_mem = instance.memlist[end:end+W]
            
            after = time()
            all = after-start
            if flag : 
                print(f' iter = {iters} the time consuming in auto arima is {(all)}s and  end is { end} instance  id is {inc_id}  the new mape = {newmape}' )
            forecast_mem[inc_id] =next_mem
            forecast_cpu[inc_id] = next_cpu
            iters += 1;
        after =  time()
        #print(f'the time consuming in auto arima is {(after-startPredict)}  and  end is { end} ')  
        inc_cpu_pre = np.array([predicts for inc_id,predicts in forecast_cpu.items()])
        inc_mem_pre = np.array([predicts  for inc_id,predicts  in forecast_mem.items()])
        inc_cpu_actual = np.array([inc.cpulist[end:end+W] for inc_id,inc in self.cluster.instances.items()])
        inc_mem_actual = np.array([inc.memlist[end:end+W] for inc_id,inc in self.cluster.instances.items()])

        N = inc_cpu_pre.shape[0]
        M = self.cluster.t_0.shape[1]
        #TODO or the schedule use
        start = time()
        
        if inc_cpu_pre.shape[1] <= W:
            placement,cost_min = sxy.SchedulePolicy( 10, 10, W, u=0.8, v=0.2, x_t0=self.cluster.t_0.copy(), N=N, cpu=inc_cpu_pre,mem=inc_mem_pre, M=M, CPU_MAX=50, MEM_MAX=75,a=0.004,b=0.01)
            after =  time() 
            
            #1433s 23min
            #print(f'the time consuming in SXY is {(after-start)}  and  end is { end} ')
            
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
            with open('/hdd/lsh/Scheduler/metric.log','a') as f:
                f.write(f'At time {end}, eval_bal={eval_bal},eval_mig = {eval_mig}  \n\t\t the different array is = {diff} \n\t\t \
                        the time consuming in SXY is {(after-start)}. cost_min ={cost_min}  \n\t\t \
                        你要的数据：{value}   \n ')
            f.close()
            #print((placement == self.cluster.t_0).all() )
            self.cluster.update_t0(placement)
           # print('N ={0} M={1}, eval_bal = {2} eval_mig = {3}'.format(N,M,eval_bal,eval_mig))
            #print(f'the x_t0 is {self.cluster.t_0}')
            del placement
        return value
            
            
            
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
