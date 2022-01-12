class Monitor(object):
    def __init__(self, env, trigger, algorithm):
        self.env = env
        self.trigger = trigger
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def sample(self):
        for instance, instance_cpu_curve in self.simulation.instance_cpu_curves.items():
            instance.cpu = instance_cpu_curve.pop(0)
        for instance, instance_memory_curve in self.simulation.instance_memory_curves.items():
            instance.memory = instance_memory_curve.pop(0)
        return True

    def run(self):
        while not self.simulation.finished:
            self.sample()
            self.trigger(self.cluster, self.env.now)
            if self.cluster.machines_to_schedule:
                print('At', self.env.now, 'scheduler was triggered!')
                print("Machines to schedule", [machine.id for machine in self.cluster.machines_to_schedule])
                self.find_candidates()
                self.make_decision()
                yield self.env.timeout(1)
            yield self.env.timeout(15)
        print(self.env.now)
