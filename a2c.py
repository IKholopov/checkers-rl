import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from checkers import CheckersEnvironment
import checkers_swig
from checkers_swig import MakeRandomStrategy, MakeMinMaxStrategy, MakeMCSTStrategy, MakeQNNStrategy,  MakeA2CStrategy, Team_Black, Team_White
from env_pool import EnvPool
from filelock import FileLock
import os

def make_qnn_strategy(path):
    return MakeQNNStrategy(Team_Black, path)

class CheckersEnvWrapper(CheckersEnvironment):
    def __init__(self, make_strat):
        super(CheckersEnvWrapper, self).__init__(american=True)
        self.reset(make_strat)
        
    def reset(self, make_strat=None):
        if make_strat is not None:
            self.make_strat = make_strat
        super(CheckersEnvWrapper, self).reset(american=True, max_steps=1000,
                                                 black_strategy=self.make_strat())
        
    def _observation(self, img):
        """what happens to each observation"""
        img = CheckersEnvironment._observation(self, img)
        normalized = img
        normalized[img == 2] = -1
        normalized[img == 4] = -2
        normalized[img == 3] = 2
        
        return normalized
        

def make_a2c_str(path):
    lock = FileLock('.lock')
    with lock:
        return checkers_swig.MakeA2CStrategy(checkers_swig.Team_Black, path, False)

def make_strat(path):
    if path == 'rand':
        return lambda: checkers_swig.MakeRandomStrategy(np.random.randint(1000000))
    elif path[:6] == 'minmax':
        return lambda: checkers_swig.MakeMinMaxStrategy(int(path.split('_')[1]))
    elif path[:4] == 'mcst':
        if path[:9] == 'mcst_rand':
            return lambda: checkers_swig.MakeMCSTStrategy(checkers_swig.Team_Black, np.random.randint(10, 500))
        return lambda: checkers_swig.MakeMCSTStrategy(checkers_swig.Team_Black, int(path.split('_')[1]))
    elif path[:3] == 'qnn':
        return lambda: checkers_swig.MakeQNNStrategy(checkers_swig.Team_Black, path)
    elif path[:3] == 'a2c':
        return lambda: make_a2c_str(path)
    assert False, 'invalid strat ' + path

def make_env(path):
    env = CheckersEnvWrapper(make_strat(path))
    env.reset()
    return env

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1


class A2CAgent(nn.Module):
    def __init__(self, action_state_shape, reuse=False):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()

        self.action_state_shape = action_state_shape
        self.conv1_state = nn.Conv2d(1, 16, 3, 2)
        self.conv1 = nn.Conv2d(self.action_state_shape[0], 16, 3, 2)
        self.conv2_state = nn.Conv2d(16, 32, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        out_conv_shape = conv2d_size_out(conv2d_size_out(self.action_state_shape[1], 3, 2), 3, 2),                           conv2d_size_out(conv2d_size_out(self.action_state_shape[2], 3, 2), 3, 2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(32 * out_conv_shape[0] * out_conv_shape[1], 256)
        self.linear1_state = nn.Linear(32 * out_conv_shape[0] * out_conv_shape[1], 256)

        self.logits = nn.Linear(256, 1)
        self.state_value = nn.Linear(256, 1)

    def conv(self, state_t):
        # in: states (batch_size, actions, 2, (w, h))
        # out: logits (batch_size, actions)
        batch_size = state_t.size(0)
        actions = state_t.size(1)
        x = state_t.view(-1, *self.action_state_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = x.view(batch_size, actions, -1)
        x = F.relu(self.linear1(x))
        logits = self.logits(x)

        st = state_t[:, 0, 0]
        x = st.view(-1, 1, *self.action_state_shape[1:])
        x = F.relu(self.conv1_state(x))
        x = F.relu(self.conv2_state(x))
        x = self.flatten(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.linear1_state(x))
        state_value = self.state_value(x)

        return logits.squeeze(2), state_value.squeeze(1)

    def forward(self, states, batch_lengths):
        # in: states (batch_size, actions, 2, (w, h))
        #     batch_lengths (batch_size)
        # out: log_pred (batch_pred, actions)
        #      state_value(batch_pred,)
        batch_size = states.size(0)
        max_action_length = states.size(1)
        logits, state_value = self.conv(states)
        mask = torch.arange(max_action_length).repeat(batch_size, 1) >= batch_lengths.unsqueeze(1)
        logits -= mask.float() * 100000
        log_pred = F.log_softmax(logits, dim=1)

        return log_pred, state_value

    def batch_lengths(self, actions_list):
        return list(map(len, actions_list))

    def create_states_batch(self, states, actions_list, max_length = None):
        batch_lengths = self.batch_lengths(actions_list)
        if max_length is None:
            max_length = max(batch_lengths)
        states_list = []
        for state, actions in zip(states, actions_list):
            this_state_actions = [[state, action] for action in actions]
            this_state_actions += [[state, np.zeros_like(state)] for _ in range(max_length - len(actions))]
            states_list.append(np.array(this_state_actions))
        states = np.array(states_list)
        return states, batch_lengths

    def sample_actions(self, agent_outputs, batch_lengths):
        max_length = max(batch_lengths)
        batch_lengths = torch.tensor(batch_lengths)

        log_logits, state_values = agent_outputs
        batch_size = log_logits.size(0)
        mask = torch.arange(max_length).repeat(batch_size, 1) >= batch_lengths.unsqueeze(1)

        probs = torch.exp(log_logits)
        probs -= mask.float() * 100000
        probs = torch.clamp(probs, 0)

        mask_zero = (batch_lengths == 0).repeat(max_length, 1).transpose(0, 1)
        probs += mask_zero.float()

        return torch.multinomial(probs, 1)[:, 0].detach().numpy()

    def step(self, states, actions_list):
        states, batch_lengths = self.create_states_batch(states, actions_list)
        states_t = torch.tensor(states, dtype=torch.float32)
        batch_lengths_t = torch.tensor(batch_lengths)
        log_pred_t, values_t = self.forward(states_t, batch_lengths_t)
        return log_pred_t.detach(), values_t.detach()


def evaluate(agent, env, n_games=1):
    game_rewards = []
    for _ in range(n_games):

        env.reset()
        observation = np.array([env.observation()])

        total_reward = 0
        while True:
            action_list = [env.current_possible_actions_values()]
            batch_length = agent.batch_lengths(action_list)
            readouts = agent.step(observation, action_list)
            action = agent.sample_actions(readouts, batch_length)[0]
            observation, reward, done, info = env.step(action)
            observation = np.array([observation])
            reward = reward[0]

            total_reward += reward
            if done:
                break

        game_rewards.append(total_reward)
    return game_rewards


def to_one_hot(y, n_dims=None):
    y_tensor = y.to(dtype=torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def train_on_rollout(agent, opt, states, actions, possible_actions, rewards, is_not_done, gamma=0.99):

    # shape: [batch_size, time, c, h, w]
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.int64)  # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # shape: [batch_size, time]
    is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32)  # shape: [batch_size, time]
    rollout_length = rewards.shape[1] - 1

    n_actions = np.max(list(map(len, possible_actions.reshape(-1))))

    log_probs = []
    state_values = []
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        possible_actions_t = possible_actions[:, t]

        states_np, batch_lengths = agent.create_states_batch(obs_t.detach().numpy(),
                                                            possible_actions_t, max_length=n_actions)
        states_t = torch.tensor(states_np, dtype=torch.float32)
        batch_lengths_t = torch.tensor(batch_lengths)
        log_probs_t, values_t = agent.forward(states_t, batch_lengths_t)

        log_probs.append(log_probs_t)
        state_values.append(values_t)

    # (batch_size, seq, n_actions)
    log_probs = torch.stack(log_probs, dim=1)
    # (batch_size, seq, n_actions)
    state_values = torch.stack(state_values, dim=1)
    probas = torch.exp(log_probs)
    logprobas = log_probs

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

    J_hat = 0
    value_loss = 0

    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t]                                # current rewards
        V_t = state_values[:, t]
        V_next = state_values[:, t + 1].detach()           # next state values
        logpi_a_s_t = logprobas_for_actions[:, t]
        is_not_done_t = is_not_done[:, t].detach()

        cumulative_returns = G_t = r_t + is_not_done_t * gamma * cumulative_returns
        # print('G_t ', G_t)
        value_loss += ((r_t + is_not_done_t * gamma * V_next - V_t)**2).mean()
        # print('VL ', (r_t + is_not_done_t * gamma * V_next - V_t)**2)
        advantage = (cumulative_returns - V_t)
        # print('adv ', advantage)
        advantage = advantage.detach()

        J_hat += (logpi_a_s_t * advantage).mean()

    entropy_reg = (- logprobas * probas).sum(dim=(1,2)).mean()

    loss = -J_hat / rollout_length +\
        value_loss / rollout_length +\
           -0.01 * entropy_reg

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.data.numpy(), (J_hat / rollout_length).data.numpy(), (value_loss / rollout_length).data.numpy(), entropy_reg.data.numpy()

def train_against(agent, opt, pool, n_rolls, eval_env, yield_every = 100, roll_size = 50):
    rewards_history = []
    j_hat_hist = []
    vl_hist = []
    ent_hist = []
    j_hat_l = []
    vl_l = []
    ent_l = []
    for i in trange(n_rolls):
        rollout_obs, rollout_actions, rollout_possible_actions, rollout_rewards, rollout_mask = pool.interact(roll_size)
        _, j_hat, vl, ent = train_on_rollout(agent, opt, rollout_obs, rollout_actions, rollout_possible_actions, rollout_rewards, rollout_mask)
        j_hat_l.append(j_hat)
        vl_l.append(vl)
        ent_l.append(ent)
        if i % yield_every == 0 and i > 0:
            j_hat_hist.append(np.mean(j_hat_l))
            vl_hist.append(np.mean(vl_l))
            ent_hist.append(np.mean(ent_l))
            rewards_history.append(np.mean(evaluate(agent, eval_env, n_games=20)))
            yield rewards_history, j_hat_hist, vl_hist, ent_hist
    return rewards_history, j_hat_hist, vl_hist, ent_hist

def train_against_strats(agent, opt, strats, eval_strat, n_rolls, n_sync):
    def f(x):
        return lambda: make_env(x)
    pool = EnvPool(agent, [f(x) for x in strats])
    eval_env = make_env(eval_strat)
    return train_against(agent, opt, pool, n_rolls, eval_env, n_sync)

def save(agent, path):
    st = np.zeros((8, 8), dtype=float)
    batch, lengths = agent.create_states_batch([st], [[st]])
    traced_script_module = torch.jit.trace(agent, (torch.tensor(batch, dtype=torch.float32), torch.tensor(lengths)))
    traced_script_module.save(path)
