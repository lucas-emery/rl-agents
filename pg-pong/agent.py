import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import sleep


H = 50
D = 25600
learn_rate = 1e-3
discount_factor = 0.99

resume = True
render = False


def main():
    env = gym.make('Pong-v0')
    observation = env.reset()

    if resume:
        with open("model.p", "rb") as f:
            W = pickle.load(f)
    else:
        w1 = np.random.rand(H, D) / math.sqrt(D)    # Xavier init
        w2 = np.random.rand(1, H) / math.sqrt(H)

        W = [w1, w2]

    ep = 0
    while True:
        xs, hs = [], []
        ds_log_prob, rs = [], []
        total_reward = 0
        done = False
        prev_x = None
        while not done:
            if render:
                env.render()
                sleep(1/60)


            cur_x = preprocess(observation)
            x = cur_x + prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x

            prob = forward_pass(x, W, xs, hs)

            action = 4 if np.random.uniform() < prob else 3

            label = 1 if action == 4 else 0     # Fake label for calculating the grad of log(policy) wrt the output layer (log_prob)

            ds_log_prob.append(label - prob)  # If action == UP then grad = 1 - prob else grad = -prob

            observation, reward, done, info = env.step(action)

            rs.append(reward)
            total_reward += reward

        print("Finished episode {0}. Total reward: {1}".format(ep, total_reward))

        #  We stack all the data from the episode, but this transposes all the vectors so the backprop math is all transposed now
        xs = np.vstack(xs)
        hs = np.vstack(hs)
        ds_log_policy = np.vstack(ds_log_prob)
        rs = np.vstack(rs)

        train(W, xs, hs, ds_log_policy, rs)

        ep += 1
        if ep % 100 == 0:
            with open("model.p", "wb") as f:
                pickle.dump(W, f)

        observation = env.reset()


def reshape(observation):
    return np.ravel(observation[34:-16, :, 0])

def preprocess(observation):
    x = np.ravel(observation[34:-16, :, 0])
    x[x == 144] = 0
    x[x == 109] = 0
    x[x != 0] = 1
    return x


def forward_pass(x, w, xs, hs):
    xs.append(x)

    h = np.dot(w[0], x)
    h[h < 0] = 0

    hs.append(h)

    log_prob = np.dot(w[1], h)
    return sigmoid(log_prob)


def train(w, xs, hs, ds_log_prob, rs):
    calculate_rewards(rs)

    rs -= np.mean(rs)
    std = np.std(rs)
    if std != 0:
        rs /= std

    ds_log_prob *= rs     # Modulate gradient with reward

    dw = backward_pass(w, xs, hs, ds_log_prob)

    for i in range(len(w)):
        w[i] += learn_rate * dw[i]


def backward_pass(w, xs, hs, ds_log_prob):
    dw1 = np.dot(ds_log_prob.T, hs)

    dhs = np.dot(ds_log_prob, w[1])
    dhs[hs == 0] = 0

    dw0 = np.dot(dhs.T, xs)

    return [dw0, dw1]


def calculate_rewards(rs):
    reward = 0
    for i in reversed(range(len(rs))):
        if rs[i] == 0:
            reward *= discount_factor
            rs[i] = reward
        else:
            reward = np.copy(rs[i])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    main()
