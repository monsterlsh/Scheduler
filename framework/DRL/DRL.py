import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        self.current_trajectory = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, inst in valid_pairs:
            # TODO Add features (Done)
            features.append([machine.cpu, machine.memory, len(machine.instances)] + self.features_extract_func(inst))
        features = self.features_normalize_func(features)
        return features

    def __call__(self, cluster, clock):
        machines = cluster.machines.values()
        instances_to_reschedule = cluster.instances_to_reschedule
        all_candidates = []

        for machine in machines:
            for inst in instances_to_reschedule:
                if machine.accommodate_w(inst):
                    all_candidates.append((machine, inst))

        if len(all_candidates) == 0:
            self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            return None, None
        else:
            features = self.extract_features(all_candidates)
            features = tf.convert_to_tensor(features, dtype=np.float32)
            logits = self.agent.policynet(features)
            pair_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1).numpy()[0]

            node = Node(features, pair_index, 0, clock)
            self.current_trajectory.append(node)

        return all_candidates[pair_index]