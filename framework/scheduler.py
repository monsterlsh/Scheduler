import logging
import random

class Scheduler(object):
    
    def __init__(self, env, algorithm,schedulePolicy):
        self.env = env
        self.algorithm_ff = algorithm
        self.algorithm_schedule = schedulePolicy
        self.simulation = None
        self.cluster = None
        self.widowsfilename = 'D:\Data\workplace\Github\scheduler\\framework\log\scheduler.log'
    

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster
    
    def sample(self):
        for instance, instance_cpu_curve in self.simulation.instance_cpu_curves.items():
            instance.cpu = instance_cpu_curve.pop(0)
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
        instancs_tosc = len( self.cluster.instances_to_reschedule)
    
    def make_decision2(self):
        for instance in self.cluster.instances_to_reschedule:
            machine = self.algorithm_ff(self.cluster, self.env.now)[0]
            # pass # TODO reschedule instance
            print(f'At {self.env.now}, {machine.id} choose {instance.id} ')
            if instance.machine is not None:
                print()
                print(instance.machine.to_schedule)
            machine.push(instance)
        instancs_tosc = len( self.cluster.instances_to_reschedule)
        self.cluster.instances_to_reschedule.clear()
    
    def find_candidates(self):
        instances_to_reschedule_list = [inst 
            for machine in self.cluster.machines_to_schedule 
                for inst in machine.instances.values()]
        #test
        mac = {macid:[ k for k,v in machine.instances.items()] 
            for macid,machine in self.cluster.machines.items() }
        for k,v in mac.items():
            print('test every mc',k,v)

        instances_to_reschedule = set()
        instances_to_reschedule.update(instances_to_reschedule_list)
        
        #logging.info('Instance to be scheduled:',insid_to_schedule)
        print('Instance to be scheduled:',[ins.id for ins in instances_to_reschedule])

        '''choose which instance to schduler
            random first select 
        '''
        #self.algorithm_ff.ChooseWhcihInstance(self.simulation,self.cluster,instances_to_reschedule)
        for inst in instances_to_reschedule:
            #if inst.machine.to_schedule :#and not self.simulation.trigger.isOverhead(inst.machine) :
            inst.machine.to_schedule = False
            #self.cluster.machines_to_schedule.remove(inst.machine)
            #没有pop掉
            inst.machine.pop(inst.id)
            inst.machine = None        
        self.cluster.instances_to_reschedule = instances_to_reschedule
   
    def everytime(self):
        self.sample()
        self.simulation.trigger(self.cluster, self.env.now)
        if self.cluster.machines_to_schedule:
            print('At', self.env.now, 'scheduler was triggered!')
            print("Machines to schedule", [machine.id for machine in self.cluster.machines_to_schedule])
            #logging.info('At', self.env.now, 'scheduler was triggered!')
            #logging.info("Machines to schedule", [machine.id for machine in self.cluster.machines_to_schedule])
            self.find_candidates()
            self.make_decision2()
            yield self.env.timeout(1)
    
    def period_make_decision(self,first,end):
        inc_cpuNow = {inc_id:inc.cpulist[self.env.now] for inc_id,inc in self.cluster.instances.items()}
        print(f'inc_cpuNow\'s type is {type(inc_cpuNow)}')
        inc_cpuNow_sort = sorted(inc_cpuNow.items(),key=lambda x:x[1])
        print(f'after sorted inc_cpuNow\'s type is {type(inc_cpuNow)}')
        inc_cpuHistory = {inc_id:inc.cpulist[first:end+1] for inc_id,inc in self.cluster.instances.items()}
        machine_ids = [v.id for k,v in self.cluster.machines.items()]
        for mac_id in machine_ids:
            self.cluster.machines[mac_id].instances.clear()
        for inc in inc_cpuNow_sort:
            #print(f'instance {instance_config.id} \'s cpu is {instance_config.cpu}')
            inc_id = inc[0]
            mac_set = set()
            mac_num = len(self.cluster.machines.items())
            while True:
                machine_id = random.randint(0,mac_num-1)
                mac_set.add(machine_id)
                machine = self.cluster.machines.get(machine_id, None)
                assert machine is not None
                
                if machine.add_period_inc(inc_id,inc_cpuNow.get(inc_id),self.cluster.instances):
                    print(f'In period {self.env.now}, instance {inc_id} choose machine {machine_id}')
                    break
                if len(mac_set) == mac_num:
                    machine.add_period_inc(inc_id,inc_cpuNow.get(inc_id),self.cluster.instances,True)
                    break
                else:
                    print(f'{self.env.now} the remain capacity of machine_{machine_id} is {machine.cpu}')
    
    def periodSchedule(self,t,first,end):
        if end-first+1 == t:
            self.algorithm_schedule(self.cluster,self.env,first,end)
            return True
        return False
    
    def run(self):
        last = 0
        end = 0
        window = 10
        while not self.simulation.finished(True):
            #self.everytime()
            print(f'time go on {self.env.now}')
            end = self.env.now
           
            if self.periodSchedule(window,last,end):
                print(f'period schedule on time {self.env.now} ')
                last = end
            yield self.env.timeout(1)
        #logging.info('now finish time:',self.env.now)
        print('now finish time:',self.env.now)
