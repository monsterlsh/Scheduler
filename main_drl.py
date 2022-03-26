import os
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

import warnings
warnings.filterwarnings("ignore")

sys.path.append('..')

from data.loader import InstanceConfigLoader
from framework.instance import InstanceConfig
from framework.machine import MachineConfig

from framework.episode import Episode
from framework.trigger import ThresholdTrigger

from framework.DRL.agent import Agent
from framework.DRL.DRL import RLAlgorithm
from framework.DRL.policynet import PolicyNet
from framework.DRL.reward_giver import AverageCompletionRewardGiver, MakespanRewardGiver,SxyRewardGiver
from framework.DRL.utils import features_extract_func, features_normalize_func, multiprocessing_run

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_random_seed(41)
#tf.random.set_seed(41)
# ************************ Data loading Start ************************
# for cloudsimpy
machines_number = 5
# jobs_len = 10
# jobs_csv = '../jobs_files/jobs.csv'
# machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
# csv_reader = CSVReader(jobs_csv)
# jobs_configs = csv_reader.generate(0, jobs_len)

# for partitionschduler
machine_configs = {
   0: MachineConfig(0, 12, 20, 20),
    1:MachineConfig(1, 21, 25, 25)
}  # MachineConfigLoader('./data/machine_resources.a.csv'):id+cpu+mem+disk

instance_configs = {
    0:InstanceConfig(0, 0, 3, 5, 5, [3, 6, 5], [5, 5, 5]),
    1:InstanceConfig(0, 1, 3, 5, 5, [3, 2, 1], [5, 5, 5]),
   2: InstanceConfig(0, 2, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
   3: InstanceConfig(1, 3, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
   4: InstanceConfig(1, 4, 5, 5, 5, [5, 5, 5], [5, 5, 5])
}  # InstanceConfigLoader('./data/output_instance_deployed_a.csv')

# for alibaba trace
# machines_number = 1313
# machine_configs = [MachineConfig(i, 64, 0, 0) for i in range(machines_number)]
# vm_cpu_request_file = 'D:/cloudsim/modules/cloudsim-examples/src/main/resources/configuration/instance_plan_cpu_all_withoutid.csv'
# vm_machine_id_file = 'D:/cloudsim/modules/cloudsim-examples/src/main/resources/configuration/instance_machind_id_all.csv'
# vm_cpu_utils_folder = 'D:/cloudsim/modules/cloudsim-examples/src/main/resources/workload/alibaba2017/instance_all'
# instance_configs = InstanceConfigLoader(vm_cpu_request_file, vm_machine_id_file, vm_cpu_utils_folder)
# ************************ Data loading End ************************

# ************************ Parameters Setting Start ************************
n_iter = 100
n_episode = 12

policynet = PolicyNet(5)
reward_giver =  AverageCompletionRewardGiver() # MakespanRewardGiver(-1)
features_extract_func = features_extract_func
features_normalize_func = features_normalize_func

name = '%s-%s-m%d' % (reward_giver.name, policynet.name, machines_number)
model_dir = './agents/%s' % name
# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

agent = Agent(name, policynet, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
              model_save_path='%s/model.ckpt' % model_dir)


for itr in range(n_iter):
    tic = time.time()
    print("********** Iteration %i ************" % itr)
    processes = []

    manager = Manager()
    trajectories = manager.list([])
    makespans = manager.list([])
    average_completions = manager.list([])
    average_slowdowns = manager.list([])
    for i in range(n_episode):
        if i==12:
            print()
        algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                                features_normalize_func=features_normalize_func)
        trigger = ThresholdTrigger()
        episode = Episode(machine_configs, instance_configs, trigger, algorithm, None)
        algorithm.reward_giver.attach(episode.simulation)
        p = Process(target=multiprocessing_run,args=(episode, trajectories, makespans))
        #episode.run()
        processes.append(p)
        print('i=',i)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    agent.log('makespan', np.mean(makespans), agent.global_step)

    toc = time.time()

    print(np.mean(makespans), toc - tic)

    all_observations = []
    all_actions = []
    all_rewards = []
    #print('\tafter multiprocess: ',trajectories,makespans)
    for trajectory in trajectories:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)

        all_observations.append(observations)
        all_actions.append(actions)
        all_rewards.append(rewards)

    all_q_s, all_advantages = agent.estimate_return(all_rewards)

    agent.update_parameters(all_observations, all_actions, all_advantages)

agent.save()