#!/usr/bin/env python3

""" Trains an agent with (stochastic) Policy Gradients on Pong. """
import numpy as np
import pickle
import gym

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
A = 3  # number of output actions
try:
    model = pickle.load(open('pong-model.p', 'rb'))  # resume from previous checkpoint
    print(f"model loaded from pong-model.p")
except FileNotFoundError:
    # Xavier initialization
    model = {'W1': np.random.randn(H, D) / np.sqrt(D),
             'W2': np.random.randn(A, H) / np.sqrt(H)}
    print(f"new model is initialized")

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make("Pong-v0")  # action 1: not move, 2: up, 3: down
observation = env.reset()
prev_x = None  # used in computing the difference frame
inputs, hiddens, dlogits, rewards = [], [], [], []
episode_number = 0

while True:
    env.render()

    cur_x = prepro(observation)
    input = cur_x - prev_x if prev_x is not None else cur_x
    prev_x = cur_x

    # forward
    inputs.append(input)
    hidden = model['W1'].dot(input)
    hidden[hidden < 0] = 0
    hiddens.append(hidden)
    logit = model['W2'].dot(hidden)
    prob = np.exp(logit) / np.sum(np.exp(logit))

    # action
    roll_out = np.random.uniform()
    if roll_out < prob[0]:
        action = 1
    elif roll_out < prob[0] + prob[1]:
        action = 2
    else:
        action = 3
    observation, reward, done, info = env.step(action)
    rewards.append(reward)

    # dlog(prob(action)) / dlogit, same as computing d_cross_entropy / d_logit
    dlogit = -prob
    dlogit[action - 1] += 1
    dlogits.append(dlogit)

    if done:
        episode_number += 1
        episode_inputs = np.vstack(inputs)
        episode_hiddens = np.vstack(hiddens).T
        episode_dlogits = np.vstack(dlogits).T
        episode_discounted_rewards = discount_rewards(np.array(rewards))
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        episode_discounted_rewards -= np.mean(episode_discounted_rewards)
        episode_discounted_rewards /= np.std(episode_discounted_rewards)

        # policy gradient
        episode_dlogits *= episode_discounted_rewards

        # back-propagate w.r.t entire trajectory
        dW2 = episode_dlogits.dot(episode_hiddens.T)
        episode_dhiddens = model['W2'].T.dot(episode_dlogits)
        episode_dhiddens[episode_hiddens <= 0] = 0
        dW1 = episode_dhiddens.dot(episode_inputs)
        grad_buffer['W1'] += dW1
        grad_buffer['W2'] += dW2

        if episode_number % batch_size == 0:
            # rmsprop
            for w, g in grad_buffer.items():
                rmsprop_cache[w] = decay_rate * rmsprop_cache[w] + (1 - decay_rate) * g * g
                model[w] += learning_rate * g / (np.sqrt(rmsprop_cache[w]) + 1e-5)
            grad_buffer['W1'] = np.zeros_like(grad_buffer['W1'])
            grad_buffer['W2'] = np.zeros_like(grad_buffer['W2'])

        if episode_number % 50 == 0:
            pickle.dump(model, open('pong-model.p', 'wb'))
        observation = env.reset()  # reset env
        prev_x = None

    if reward == 1.:
        print(f"episode {episode_number}: win :-)")
    elif reward == -1.:
        print(f"episode {episode_number}: lose !!")
