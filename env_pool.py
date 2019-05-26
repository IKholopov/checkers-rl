"""
A thin wrapper for openAI gym environments that maintains a set of parallel games and has a method to generate
interaction sessions given agent one-step applier function.
"""

import numpy as np

# A whole lot of space invaders
class EnvPool(object):
    def __init__(self, agent, make_envs):
        """
        A special class that handles training on multiple parallel sessions
        and is capable of some auxilary actions like evaluating agent on one game session (See .evaluate()).

        :param agent: Agent which interacts with the environment.
        :param make_env: Factory that produces environments OR a name of the gym environment.
        :param n_games: Number of parallel games. One game by default.
        :param max_size: Max pool size by default (if appending sessions). By default, pool is not constrained in size.
        """
        # Create atari games.
        self.agent = agent
        self.make_envs = make_envs
        n_parallel_games = len(self.make_envs)
        self.envs = [self.make_envs[i]() for i in range(n_parallel_games)]

        # Initial observations.
        [env.reset() for env in self.envs]
        self.prev_observations = [env.observation() for env in self.envs]

        # Whether particular session has just been terminated and needs restarting.
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100, verbose=False):
        """Generate interaction sessions with ataries (openAI gym atari environments)
        Sessions will have length n_steps. Each time one of games is finished, it is immediately getting reset
        and this time is recorded in is_alive_log (See returned values).

        :param n_steps: Length of an interaction.
        :returns: observation_seq, action_seq, reward_seq, is_alive_seq
        :rtype: a bunch of tensors [batch, tick, ...]
        """

        def env_step(i, action):
            new_observation, cur_reward, is_done, info = self.envs[i].step(action)
            cur_reward = cur_reward[0]
            alive = True
            if is_done:
                # Game ends now, will finalize on next tick.
                if verbose:
                    print("env %i done" % i)
                alive = False
                self.envs[i].reset()
                new_observation = self.envs[i].observation()
                if verbose:
                    print("env %i reloaded" % i)

                # note: is_alive=True in any case because environment is still alive (last tick alive) in our notation.
            return new_observation, cur_reward, alive, info

        history_log = []

        for i in range(n_steps - 1):
            possible_actions = [env.current_possible_actions_values() for env in  self.envs]
            batch_lengths = self.agent.batch_lengths(possible_actions)
            readout = self.agent.step(self.prev_observations, possible_actions)
            actions = self.agent.sample_actions(readout, batch_lengths)

            new_observations, cur_rewards, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))

            # Append data tuple for this tick.
            history_log.append((self.prev_observations, actions, possible_actions, cur_rewards, is_alive))

            self.prev_observations = new_observations
        
        #add last observation
        dummy_actions = [0] * len(self.envs)
        dummy_possible = [[self.envs[0].observation()] for _ in  range(len(self.envs))]
        dummy_rewards = [0] * len(self.envs)
        dummy_mask = [1] * len(self.envs)
        history_log.append((self.prev_observations, dummy_actions, dummy_possible, dummy_rewards, dummy_mask))

        # cast to numpy arrays, transpose from [time, batch, ...] to [batch, time, ...]
        history_log = [np.array(tensor).swapaxes(0, 1) for tensor in zip(*history_log)]
        observation_seq, action_seq, possible_action_seq, reward_seq, is_alive_seq = history_log

        return observation_seq, action_seq, possible_action_seq, reward_seq, is_alive_seq
