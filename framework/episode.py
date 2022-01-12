import simpy
from .simulation import Simulation


class Episode(object):
    def __init__(self, machine_configs, instance_configs, trigger, algorithm, event_file):
        self.simulation = Simulation(machine_configs, instance_configs, trigger, algorithm)

    def run(self):
        self.simulation.run()