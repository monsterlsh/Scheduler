import time
import numpy as np
import tensorflow as tf


def features_extract_func(instance):
    return [instance.cpu, instance.memory]


def features_normalize_func(x):
    y = np.array(x)
    return y



def multiprocessing_run(episode, trajectories, makespans):
    np.random.seed(int(time.time()))
    tf.random.set_random_seed(time.time())
    episode.run()
    # print('multiprocessing_run:')
    # for tra in episode.simulation.monitor.algorithm.current_trajectory: 
    #     print(f'\t\t obs = {tra.observation} action = {tra.action} reward={tra.reward} clock={tra.clock}')
    trajectories.append(episode.simulation.monitor.algorithm.current_trajectory)
    makespans.append(episode.simulation.env.now)
    # print(episode.simulation.env.now)