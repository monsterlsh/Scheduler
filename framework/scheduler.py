import logging
import random
class Scheduler(object):
    
    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None
        self.widowsfilename = 'D:\Data\workplace\Github\scheduler\\framework\log\scheduler.log'
        '''logging.basicConfig(level=logging.INFO,
                            filename=self.widowsfilename,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')'''

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster
    
    def sample(self):
        for instance, instance_cpu_curve in self.simulation.instance_cpu_curves.items():
            instance.cpu = instance_cpu_curve.pop(0)
        return True

    def make_decision(self):
        while True:
            machine, instance = self.algorithm(self.cluster, self.env.now)
            if machine is None or instance is None:
                break
            else:
                # pass # TODO reschedule instance
                if instance in self.cluster.instances_to_reschedule:
                    self.cluster.instances_to_reschedule.remove(instance)
                print(f'At {self.env.now}, {machine.id} choose {instance.id} ')
                machine.push(instance)
        instancs_tosc = len( self.cluster.instances_to_reschedule)
    def make_decision2(self):
        for instance in  self.cluster.instances_to_reschedule:
            machine = self.algorithm(self.cluster, self.env.now)[0]
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
        #self.algorithm.ChooseWhcihInstance(self.simulation,self.cluster,instances_to_reschedule)
        for inst in instances_to_reschedule:
            #if inst.machine.to_schedule :#and not self.simulation.trigger.isOverhead(inst.machine) :
            inst.machine.to_schedule = False
            #self.cluster.machines_to_schedule.remove(inst.machine)
            #没有pop掉
            inst.machine.pop(inst.id)
            inst.machine = None        
        self.cluster.instances_to_reschedule = instances_to_reschedule
   

    def run(self):
        while not self.simulation.finished:
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
            else:
                print('no schedule')
            #print(self.cluster.structure)
            yield self.env.timeout(1)
        #logging.info('now finish time:',self.env.now)
        print('now finish time:',self.env.now)
