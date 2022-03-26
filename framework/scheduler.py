import logging
import random
from time import time
from framework import sandpiper as sand
import numpy as np


class Scheduler(object):

    def __init__(self, env, algorithm, schedulePolicy):
        self.env = env
        self.algorithm_ff = algorithm
        self.algorithm_schedule = schedulePolicy
        self.simulation = None
        self.cluster = None
        #self.widowsfilename = 'D:\Data\workplace\Github\scheduler\\framework\log\scheduler.log'

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def sample(self):
        cpu_list = np.zeros(shape=len(self.cluster.instances.keys()))
        mem_list = np.zeros(shape=len(self.cluster.instances.keys()))
        for k, mac in self.cluster.machines.items():
            inc = mac.instances

            inc_arry = []
            for v in inc.values():
                v.cpu = v.cpulist.pop(0)
                v.mem = v.memlist.pop(0)
                inc_arry.append(v.cpu)
                cpu_list[v.id] = v.cpu
                mem_list[v.id] = v.mem
            inc_arry = np.asarray(inc_arry)
            #print(f'mac_{k} has {len(inc)} instances its sum cpu is {np.sum(inc_arry)}')
        self.cluster.update_cpu_mem(cpu_list, mem_list)

        return True

    def make_decision(self):
        while True:
            machine, instance = self.algorithm_ff(self.cluster, self.env.now)
            if machine is None or instance is None:
                break
            else:
                # TODO reschedule instance
                if instance in self.cluster.instances_to_reschedule:
                    self.cluster.instances_to_reschedule.remove(instance)
                print(f'At {self.env.now}, {machine.id} choose {instance.id} ')
                machine.push(instance)
        instancs_tosc = len(self.cluster.instances_to_reschedule)

    def make_decision2(self):
        for instance in self.cluster.instances_to_reschedule:
            machine = self.algorithm_ff(self.cluster, self.env.now)[0]
            # pass # TODO reschedule instance
            print(f'At {self.env.now}, {machine.id} choose {instance.id} ')
            if instance.machine is not None:
                print()
                print(instance.machine.to_schedule)
            machine.push(instance)
        instancs_tosc = len(self.cluster.instances_to_reschedule)
        self.cluster.instances_to_reschedule.clear()

    def find_candidates(self):
        instances_to_reschedule_list = [inst
                                        for machine in self.cluster.machines_to_schedule
                                        for inst in machine.instances.values()]
        # test
        mac = {macid: [k for k, v in machine.instances.items()]
               for macid, machine in self.cluster.machines.items()}
        for k, v in mac.items():
            print('test every mc', k, v)

        instances_to_reschedule = set()
        instances_to_reschedule.update(instances_to_reschedule_list)

        #logging.info('Instance to be scheduled:',insid_to_schedule)
        print('Instance to be scheduled:', [
              ins.id for ins in instances_to_reschedule])

        '''choose which instance to schduler
            random first select 
        '''
        # self.algorithm_ff.ChooseWhcihInstance(self.simulation,self.cluster,instances_to_reschedule)
        for inst in instances_to_reschedule:
            # if inst.machine.to_schedule :#and not self.simulation.trigger.isOverhead(inst.machine) :
            inst.machine.to_schedule = False
            # self.cluster.machines_to_schedule.remove(inst.machine)
            # 没有pop掉
            inst.machine.pop(inst.id)
            inst.machine = None
        self.cluster.instances_to_reschedule = instances_to_reschedule

    def everytime(self):
        self.sample()
        self.simulation.trigger(self.cluster, self.env.now)
        if self.cluster.machines_to_schedule:
            print('At', self.env.now, 'scheduler was triggered!')
           # print("Machines to schedule", [machine.id for machine in self.cluster.machines_to_schedule])
            self.algorithm_ff(self.cluster, self.env.now)
            # self.find_candidates()
            # self.make_decision2()

            yield self.env.timeout(1)

    def period_make_decision(self, first, end):
        inc_cpuNow = {inc_id: inc.cpulist[self.env.now]
                      for inc_id, inc in self.cluster.instances.items()}
        print(f'inc_cpuNow\'s type is {type(inc_cpuNow)}')
        inc_cpuNow_sort = sorted(inc_cpuNow.items(), key=lambda x: x[1])
        print(f'after sorted inc_cpuNow\'s type is {type(inc_cpuNow)}')
        inc_cpuHistory = {inc_id: inc.cpulist[first:end+1]
                          for inc_id, inc in self.cluster.instances.items()}
        machine_ids = [v.id for k, v in self.cluster.machines.items()]
        for mac_id in machine_ids:
            self.cluster.machines[mac_id].instances.clear()
        for inc in inc_cpuNow_sort:
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            inc_id = inc[0]
            mac_set = set()
            mac_num = len(self.cluster.machines.items())
            while True:
                machine_id = random.randint(0, mac_num-1)
                mac_set.add(machine_id)
                machine = self.cluster.machines.get(machine_id, None)
                assert machine is not None

                if machine.add_period_inc(inc_id, inc_cpuNow.get(inc_id), self.cluster.instances):
                    print(
                        f'In period {self.env.now}, instance {inc_id} choose machine {machine_id}')
                    break
                if len(mac_set) == mac_num:
                    machine.add_period_inc(inc_id, inc_cpuNow.get(
                        inc_id), self.cluster.instances, True)
                    break
                else:
                    print(
                        f'{self.env.now} the remain capacity of machine_{machine_id} is {machine.cpu}')

    def periodSchedule(self, t, first, end):
        if end-first == t:
            self.algorithm_schedule(self.cluster, self.env, first, end)
            return True
        return False

    def run(self):
        sum = 0
        while not self.simulation.finished(True):
            # self.everytime()
            start = time()
            end = self.env.now
            value = self.algorithm_schedule(self.cluster, self.env, 0, end)
            sum += value
            after = time()
            print(f'sxy 调度决策消耗了{(after-start)}s time go on {self.env.now} sum = {sum}')
            yield self.env.timeout(1)
        print('now finish time:', self.env.now)

    def run_drl(self):
        while not self.simulation.finished(True):
            # self.everytime()
            start = time()
            end = self.env.now
            # TODO
            print(f'{self.algorithm_ff}')
            self.algorithm_ff(self.cluster, self.env)
            after = time()
            print(f'调度决策消耗了{(after-start)}s time go on {self.env.now}')
            yield self.env.timeout(1)
        print('now finish time:', self.env.now)

    def run_sand(self):
        sum = 0
        while not self.simulation.finished(True):
            self.sample()
            self.simulation.trigger(self.cluster, self.env.now)
            if len(self.cluster.machines_to_schedule) > 0:
                print('At', self.env.now, 'scheduler was triggered!')
                # print("Machines to schedule", [machine.id for machine in self.cluster.machines_to_schedule])
                sum += self.algorithm_ff(self.cluster, self.env.now)
                print(f'\t\t sum = {sum}')
                # self.find_candidates()
                # self.make_decision2()
                print(f'\t\tperiod schedule on time {self.env.now} ')
            yield self.env.timeout(1)
        print('now finish time:', self.env.now)
