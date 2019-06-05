import numpy as np

class EnvPool(object):
    def __init__(self, agent, make_envs):
        self.agent = agent
        self.make_envs = make_envs
        n_parallel_games = len(self.make_envs)
        self.envs = [self.make_envs[i]() for i in range(n_parallel_games)]

        [env.reset() for env in self.envs]
        self.prev_observations = [env.observation() for env in self.envs]
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100, verbose=False):
        def env_step(i, action):
            new_observation, cur_reward, is_done, info = self.envs[i].step(action)
            cur_reward = cur_reward[0]
            alive = True
            if is_done:
                if verbose:
                    print("env %i done" % i)
                alive = False
                self.envs[i].reset()
                new_observation = self.envs[i].observation()
                if verbose:
                    print("env %i reloaded" % i)
            return new_observation, cur_reward, alive, info

        history_log = []

        for i in range(n_steps - 1):
            possible_actions = [env.current_possible_actions_values() for env in  self.envs]
            batch_lengths = self.agent.batch_lengths(possible_actions)
            readout = self.agent.step(self.prev_observations, possible_actions)
            actions = self.agent.sample_actions(readout, batch_lengths)

            new_observations, cur_rewards, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))

            history_log.append((self.prev_observations, actions, possible_actions, cur_rewards, is_alive))

            self.prev_observations = new_observations
        
        dummy_actions = [0] * len(self.envs)
        dummy_possible = [[self.envs[0].observation()] for _ in  range(len(self.envs))]
        dummy_rewards = [0] * len(self.envs)
        dummy_mask = [1] * len(self.envs)
        history_log.append((self.prev_observations, dummy_actions, dummy_possible, dummy_rewards, dummy_mask))

        history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        observation_seq, action_seq, possible_action_seq, reward_seq, is_alive_seq = history_log

        return observation_seq, action_seq, possible_action_seq, reward_seq, is_alive_seq
