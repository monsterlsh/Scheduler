from abc import ABC, abstractmethod
import functools

class Algorithm(ABC):
    @abstractmethod
    def __call__(self, cluster, clock):
        pass


# class ThresholdFirstFitAlgorithm(Algorithm):
#     def __call__(self, cluster, clock, cpu_threshold=0.75, memory_threshold=0.75, disk_threshold=0.75):
#         instances_to_reschedule = [inst for machine in cluster.machines_to_schedule for inst in machine.instances.values()]
#         machines = cluster.machines.values()
#         for inst in instances_to_reschedule:
#             if inst.machine.to_schedule:
#                 inst.machine.to_schedule = False
#                 cluster.machines_to_schedule.remove(inst.machine)
#             inst.machine.pop(inst.id)
#         for inst in instances_to_reschedule:
#             for machine in machines:
#                 if machine.accommodate_w(inst):
#                     machine.push(inst)

class ThresholdFirstFitAlgorithm(Algorithm):
    def __call__(self, cluster, clock):
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
