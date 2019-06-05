#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import torch
import utils
import checkers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import checkers_swig
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from IPython.display import clear_output
import matplotlib.pyplot as plt
from utils import play_and_log_episode, img_by_obs
from tensorboardX import SummaryWriter
from time import gmtime, strftime

class PreprocessedCheckers(checkers.CheckersEnvironment):
    def __init__(self, american):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        checkers.CheckersEnvironment.__init__(self, american)


    def _observation(self, img):
        """what happens to each observation"""
        img = checkers.CheckersEnvironment._observation(self, img)
        normalized = img
        normalized[img == 2] = -1
        normalized[img == 4] = -2
        normalized[img == 3] = 2
        
        return normalized

def make_opts():
    return dict(american=True, black_strategy=checkers_swig.MakeMCSTStrategy(checkers_swig.Team_Black, 100))

def make_env():
    opts = make_opts()
    env = PreprocessedCheckers(opts['american'])
    env.reset(**opts)
    return env, opts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def conv2d_size_out(size, kernel_size, stride):
    """
    common use case:
    cur_layer_img_w = conv2d_size_out(cur_layer_img_w, kernel_size, stride)
    cur_layer_img_h = conv2d_size_out(cur_layer_img_h, kernel_size, stride)
    to understand the shape for dense layer's input
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


class DQNAgent(nn.Module):
    def __init__(self, action_state_shape, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.action_state_shape = action_state_shape

        self.conv1 = nn.Conv2d(self.action_state_shape[0], 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        out_conv_shape = conv2d_size_out(conv2d_size_out(self.action_state_shape[1], 3, 2), 3, 2),                           conv2d_size_out(conv2d_size_out(self.action_state_shape[2], 3, 2), 3, 2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(32 * out_conv_shape[0] * out_conv_shape[1], 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, state_t):
        """
        :param state_t: a batch of 2-frame buffers, shape = [batch_size, 2, h, w]
        """
        x = F.relu(self.conv1(state_t))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        qvalues = self.linear2(x)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == 1

        return qvalues

    def get_qvalues(self, states, actions_list):
        model_device = next(self.parameters()).device
        states_list = []
        variants_cummulative = [0]
        for state, actions in zip(states, actions_list):
            states_list += [[state, action] for action in actions]
            variants_cummulative.append(variants_cummulative[-1] + len(actions))
        states = np.array(states_list)
            
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        qvalues = qvalues.data.cpu().numpy()
        q_list = []
        for i in range(len(variants_cummulative) - 1):
            q_list.append(qvalues[variants_cummulative[i]:variants_cummulative[i + 1]])
        return q_list

    def sample_actions(self, qvalues):
        epsilon = self.epsilon
        batch_size = len(qvalues)

        random_actions = [np.random.choice(len(actions)) for actions in qvalues]
        best_actions = [actions.argmax(axis=0) for actions in qvalues]

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    rewards = []
    for _ in range(n_games):
        env.reset(**make_opts())
        s = env.observation()
        reward = 0
        for _ in range(t_max):
            actions_values = env.current_possible_actions_values()
            actions = env.possible_actions(env.env.CurrentState())
            qvalues = agent.get_qvalues([s], [actions_values])
            action = qvalues[0].argmax(axis=0)[0] if greedy else agent.sample_actions(qvalues)[0][0]
            s, r, done, _ = env.step(action)
            reward += r[0]
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, act_tp1, done):
        data = (obs_t, action, reward, obs_tp1, act_tp1, done)

        if len(self) == self._maxsize:
            self._storage = self._storage[1:]
        self._storage.append(data)

    def sample(self, batch_size):
        idxes = np.random.choice(np.arange(len(self)), batch_size)

        states, actions, rewards, next_states, next_actions, dones = [], [], [], [], [], []
        for idx in idxes:
            state, action, reward, next_state, next_action, done = self._storage[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(next_actions), np.array(dones)

def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    s = initial_state
    sum_rewards = 0

    for n in range(n_steps):
        actions_values = env.current_possible_actions_values()
        actions = env.possible_actions(env.env.CurrentState())
        q_list = agent.get_qvalues([s], [actions_values])
        a = agent.sample_actions(q_list)[0][0]
        act = actions_values[a]
        next_s, r, done, _ = env.step(a)
        exp_replay.add(s, act, r, next_s, env.current_possible_actions_values(), done)
        s = next_s
        sum_rewards += r
        if done:
            env.reset(**make_opts())
            s = env.observation()

    return sum_rewards, s

def add_actions_to_states(states, actions):
    batch = []
    indices = []
    for i, l in enumerate(actions):
        if len(l) == 0:
            l = [states[i]]
        indices.append((0 if i == 0 else indices[-1]) + len(l))
        for action in l:
            batch.append([states[i], action])
    return np.array(batch), indices

def compute_td_loss(states, actions, rewards, next_states, next_actions, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device=device):
    batch_size = states.shape[0]
    states = np.array([states, actions]).transpose((1, 0, 2, 3))
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states, next_state_idxs = add_actions_to_states(next_states, next_actions)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)

    predicted_next_qvalues = target_network(next_states)
    
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), 0]

    n_values = []
    for start, end in zip([0] + next_state_idxs[:-1], next_state_idxs):
        n_values.append(torch.max(predicted_next_qvalues[start:end], dim=0)[0])
    next_state_values = torch.cat(n_values).reshape(batch_size)
        
    target_qvalues_for_actions = gamma * next_state_values * is_not_done + rewards[:,0] # <YOUR CODE>

    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    return loss


seed = 0xbadf00d
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env, opts = make_env()
state_shape = (2, 8, 8)
state = env.observation()

agent = DQNAgent(state_shape, epsilon=1).to(device)
target_network = DQNAgent(state_shape).to(device)
target_network.load_state_dict(agent.state_dict())


buf_size = 10**4
print('Heating up replay buffer of size {}'.format(buf_size))
exp_replay = ReplayBuffer(buf_size)
for i in range(100):
    if not utils.is_enough_ram(min_available_gb=0.1):
        print("""
            Less than 100 Mb RAM available. 
            Make sure the buffer size in not too huge.
            Also check, maybe other processes consume RAM heavily.
            """
             )
        break
    play_and_record(state, agent, env, exp_replay, n_steps=10**2)
    if len(exp_replay) == buf_size:
        break
print('Finished: {} plays'.format(len(exp_replay)))

timesteps_per_epoch = 1
batch_size = 16
total_steps = 3 * 10**6
decay_steps = 10**6

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

init_epsilon = 0.4645
final_epsilon = 0.1

loss_freq = 50
refresh_target_network_freq = 5000
eval_freq = 5000

max_grad_norm = 50

n_lives = 5


# In[27]:


mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []


writer = SummaryWriter('runs/qLearning' +  strftime('%a%d%b%Y%H%M%S', gmtime()))
env.reset(**make_opts())
state = env.observation()
for step in trange(total_steps + 1):
    agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

    # play
    _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

    # train
    batch_s, batch_a, batch_r, batch_ns, batch_na, batch_done = exp_replay.sample(batch_size)

    loss = compute_td_loss(batch_s, batch_a, batch_r, batch_ns, batch_na, batch_done, agent, target_network)

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())
        grad_norm_history.append(grad_norm)

    if step % refresh_target_network_freq == 0:
        target_network.load_state_dict(agent.state_dict())

    if step % eval_freq == 0:
        torch.save(agent.state_dict(), 'qnn_checkers' + str(step) + '.pt')
        mean_rw_history.append(evaluate(
            make_env()[0], agent, n_games=3 * n_lives, greedy=True)
        )
        e = make_env()[0]
        initial_state_q_values = agent.get_qvalues(
            [e.observation()], [e.current_possible_actions_values()]
        )
        initial_state_v_history.append(np.max(initial_state_q_values))

        print("buffer size = %i, epsilon = %.5f" %
              (len(exp_replay), agent.epsilon))
        writer.add_scalar('mean_per_life', torch.tensor(mean_rw_history[-1]), step)

        assert not np.isnan(td_loss_history[-1])
        writer.add_scalar('td_loss_history', torch.tensor(utils.smoothen(td_loss_history)[-1]), step)

        writer.add_scalar('initial_state_v_history', torch.tensor(initial_state_v_history[-1]), step)

        writer.add_scalar('grad_norm_history', torch.tensor(utils.smoothen(grad_norm_history)[-1]), step)
        writer.file_writer.flush()

final_score = evaluate(
  make_env(clip_rewards=False, seed=9),
    agent, n_games=30, greedy=True, t_max=10 * 1000
) * n_lives
