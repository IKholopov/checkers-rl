#!/usr/bin/env python
# coding: utf-8

# # Deep Q-Network implementation.
# 
# __Frameworks__ - we'll accept this homework in any deep learning framework. This particular notebook was designed for pytoch, but you find it easy to adapt it to almost any python-based deep learning framework.

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

def make_env(american=True):
    env = PreprocessedCheckers(american)
    opts = dict(american=american, black_strategy=checkers.checkers_swig.MakeRandomStrategy())
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

        # Define your network body here. Please make sure agent is fully contained here
        self.conv1 = nn.Conv2d(self.action_state_shape[0], 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        out_conv_shape = conv2d_size_out(conv2d_size_out(self.action_state_shape[1], 3, 2), 3, 2),                           conv2d_size_out(conv2d_size_out(self.action_state_shape[2], 3, 2), 3, 2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(32 * out_conv_shape[0] * out_conv_shape[1], 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
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
        """
        like forward, but works on numpy arrays, not tensors
        """
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
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size = len(qvalues)

        random_actions = [np.random.choice(len(actions)) for actions in qvalues]
        best_actions = [actions.argmax(axis=0) for actions in qvalues]

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def evaluate(env, opts, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        env.reset(**opts)
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
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size

        # OPTIONAL: YOUR CODE

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, act_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize. 
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, act_tp1, done)

        if len(self) == self._maxsize:
            self._storage = self._storage[1:]
        # add data to storage
        self._storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = np.random.choice(np.arange(len(self)), batch_size)# <randomly generate batch_size integers to be used as indexes of samples >

        states, actions, rewards, next_states, next_actions, dones = [], [], [], [], [], []
        for idx in idxes:
            state, action, reward, next_state, next_action, done = self._storage[idx]# collect <s,a,r,s',done> for each indexd
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(next_actions), np.array(dones)

def play_and_record(initial_state, agent, env, opts, exp_replay, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for n in range(n_steps):# <YOUR CODE >
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
            env.reset(**opts)
            s = env.observation()

    return sum_rewards, s

def add_actions_to_states(states, actions):
    assert states.shape[0] == actions.shape[0], 'Different number of states and actions lists'
    assert len(actions.shape) == 1, 'actions should be 1-D list of np.arrays'
    batch = []
    indices = []
    for i, l in enumerate(actions):
        if len(l) == 0:
            l = [states[i]]
        indices.append((0 if i == 0 else indices[-1]) + len(l))
        for action in l:
            batch.append([states[i], action])
    return np.array(batch), indices

# ### Learning with... Q-learning
# Here we write a function similar to `agent.update` from tabular q-learning.

# Compute Q-learning TD error:
# 
# $$ L = { 1 \over N} \sum_i [ Q_{\theta}(s,a) - Q_{reference}(s,a) ] ^2 $$
# 
# With Q-reference defined as
# 
# $$ Q_{reference}(s,a) = r(s,a) + \gamma \cdot max_{a'} Q_{target}(s', a') $$
# 
# Where
# * $ Q_{target}(s',a')$ denotes q-value of next state and next action predicted by __target_network__
# * $ s, a, r, s'$ are current state, action, reward and next state respectively
# * $ \gamma$ is a discount factor defined two cells above.
#

def compute_td_loss(states, actions, rewards, next_states, next_actions, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device=device):
    batch_size = states.shape[0]
    states = np.array([states, actions]).transpose((1, 0, 2, 3))
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
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

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), 0]

    # compute V*(next_states) using predicted next q-values
    n_values = []
    for start, end in zip([0] + next_state_idxs[:-1], next_state_idxs):
        n_values.append(torch.max(predicted_next_qvalues[start:end], dim=0)[0])
    next_state_values = torch.cat(n_values).reshape(batch_size)
        
    # next_state_values = torch.max(predicted_next_qvalues, dim=1)[0] # <YOUR CODE>
    
    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = gamma * next_state_values * is_not_done + rewards[:,0] # <YOUR CODE>

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


seed = 0xbadf00d#<your favourite random seed>
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env, opts = make_env(True)
state_shape = (2, 8, 8)
state = env.observation()

agent = DQNAgent(state_shape, epsilon=1).to(device)
# В процессе были падения, но модель сохранилась
target_network = DQNAgent(state_shape).to(device)
target_network.load_state_dict(agent.state_dict())


# Buffer of size $10^4$ fits into 5 Gb RAM.
# 
# Larger sizes ($10^5$ and $10^6$ are common) can be used. It can improve the learning, but $10^4$ is quiet enough. $10^2$ will probably fail learning.
buf_size = 10**5
print('Heating up replay buffer of size {}'.format(buf_size))
exp_replay = ReplayBuffer(buf_size)
for i in range(1000):
    if not utils.is_enough_ram(min_available_gb=0.1):
        print("""
            Less than 100 Mb RAM available. 
            Make sure the buffer size in not too huge.
            Also check, maybe other processes consume RAM heavily.
            """
             )
        break
    play_and_record(state, agent, env, opts, exp_replay, n_steps=10**2)
    if len(exp_replay) == buf_size:
        break
print('Finished: {} plays'.format(len(exp_replay)))

timesteps_per_epoch = 1
batch_size = 16
total_steps = 3 * 10**6
decay_steps = 10**6

opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

init_epsilon = 1
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
env.reset(**opts)
state = env.observation()
for step in trange(total_steps + 1):
    agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

    # play
    _, state = play_and_record(state, agent, env, opts, exp_replay, timesteps_per_epoch)

    # train
    batch_s, batch_a, batch_r, batch_ns, batch_na, batch_done = exp_replay.sample(batch_size) # < sample batch_size of data from experience replay >

    loss = compute_td_loss(batch_s, batch_a, batch_r, batch_ns, batch_na, batch_done, agent, target_network) #< compute TD loss >

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    if step % loss_freq == 0:
        td_loss_history.append(loss.data.cpu().item())
        grad_norm_history.append(grad_norm)

    if step % refresh_target_network_freq == 0:
        # Load agent weights into target_network
        target_network.load_state_dict(agent.state_dict()) # <YOUR CODE >

    if step % eval_freq == 0:
        torch.save(agent.state_dict(), 'atari' + str(step) + '.pt')
        mean_rw_history.append(evaluate(
            make_env(True)[0], opts, agent, n_games=3 * n_lives, greedy=True)
        )
        e = make_env(True)[0]
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


# Agent is evaluated for 1 life, not for a whole episode of 5 lives. Rewards in evaluation are also truncated. Cuz this is what environment the agent is learning in and in this way mean rewards per life can be compared with initial state value
# 
# The goal is to get 10 points in the real env. So 2 or better 3 points in the preprocessed one will probably be enough. You can interrupt learning then.

# Final scoring is done on a whole episode with all 5 lives.

final_score = evaluate(
  make_env(clip_rewards=False, seed=9),
    agent, n_games=30, greedy=True, t_max=10 * 1000
) * n_lives


# ## How to interpret plots:
# 
# This aint no supervised learning so don't expect anything to improve monotonously. 
# * **TD loss** is the MSE between agent's current Q-values and target Q-values. It may slowly increase or decrease, it's ok. The "not ok" behavior includes going NaN or stayng at exactly zero before agent has perfect performance.
# * **grad norm** just shows the intensivity of training. Not ok is growing to values of about 100 (or maybe even 50) though it depends on network architecture.
# * **mean reward** is the expected sum of r(s,a) agent gets over the full game session. It will oscillate, but on average it should get higher over time (after a few thousand iterations...). 
#  * In basic q-learning implementation it takes about 40k steps to "warm up" agent before it starts to get better.
# * **Initial state V** is the expected discounted reward for episode in the oppinion of the agent. It should behave more smoothly than **mean reward**. It should get higher over time but sometimes can experience drawdowns because of the agaent's overestimates.
# * **buffer size** - this one is simple. It should go up and cap at max size.
# * **epsilon** - agent's willingness to explore. If you see that agent's already at 0.01 epsilon before it's average reward is above 0 - it means you need to increase epsilon. Set it back to some 0.2 - 0.5 and decrease the pace at which it goes down.
# * Smoothing of plots is done with a gaussian kernel
# 
# At first your agent will lose quickly. Then it will learn to suck less and at least hit the ball a few times before it loses. Finally it will learn to actually score points.
# 
# **Training will take time.** A lot of it actually. Probably you will not see any improvment during first **150k** time steps (note that by default in this notebook agent is evaluated every 5000 time steps).
# 
# But hey, long training time isn't _that_ bad:
# ![img](https://github.com/yandexdataschool/Practical_RL/raw/master/yet_another_week/_resource/training.png)

# ## About hyperparameters:
# 
# The task has something in common with supervised learning: loss is optimized through the buffer (instead of Train dataset). But the distribution of states and actions in the buffer **is not stationary** and depends on the policy that generated it. It can even happen that the mean TD error across the buffer is very low but the performance is extremely poor (imagine the agent collecting data to the buffer always manages to avoid the ball).
# 
# * Total timesteps and training time: It seems to be so huge, but actually it is normal for RL.
# 
# * $\epsilon$ decay shedule was taken from the original paper and is like traditional for epsilon-greedy policies. At the beginning of the training the agent's greedy policy is poor so many random actions should be taken.
# 
# * Optimizer: In the original paper RMSProp was used (they did not have Adam in 2013) and it can work not worse than Adam. For us Adam was default and it worked.
# 
# * lr: $10^{-3}$ would probably be too huge
# 
# * batch size: This one can be very important: if it is too small the agent can fail to learn. Huge batch takes more time to process. If batch of size 8 can not be processed on the hardware you use take 2 (or even 4) batches of size 4, divide the loss on them by 2 (or 4) and make optimization step after both backward() calls in torch.
# 
# * target network update frequency: has something in common with learning rate. Too frequent updates can lead to divergence. Too rare can lead to slow leraning. For millions of total timesteps thousands of inner steps seem ok. One iteration of target network updating is an iteration of the (this time approximate) $\gamma$-compression that stands behind Q-learning. The more inner steps it makes the more accurate is the compression.
# * max_grad_norm - just huge enough. In torch clip_grad_norm also evaluates the norm before clipping and it can be convenient for logging.

# ### Video


# record sessions
import gym.wrappers
env_monitor = gym.wrappers.Monitor(make_env(), directory="videos", force=True)
sessions = [evaluate(env_monitor, agent, n_games=n_lives, greedy=True) for _ in range(10)]
env_monitor.close()

# show video
from IPython.display import HTML
import os

video_names = list(
    filter(lambda s: s.endswith(".mp4"), os.listdir("./videos/")))

HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format("./videos/"+video_names[-1]))  # this may or may not be _last_ video. Try other indices
