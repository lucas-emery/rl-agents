import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import sleep


H = 50
D = 80 * 80
learn_rate = 3e-4
discount_factor = 0.99
batch_size = 10

resume = True
render = False
plot = True


def main():
    env = gym.make('Pong-v0')
    observation = env.reset()

    if resume:
        with open("model.p", "rb") as f:
            W, B, L = pickle.load(f)
    else:
        w1 = np.random.rand(H, D) / math.sqrt(D)    # Xavier init
        w2 = (np.random.rand(1, H) - 0.5) / math.sqrt(H)    # Deducted Uniform(0, 1) mean to eliminate tendency to go UP
                                                            # This is possible because negative values have no negative
        b1 = np.random.rand(H)                              # effects in this layer (sigmoid activation)
        b2 = np.zeros(1)

        W = [w1, w2]
        B = [b1, b2]
        L = []

    ep = 0
    batch_reward = 0
    while True:
        xs, hs = [], []
        ds_log_prob, rs = [], []
        ep_reward = 0
        done = False
        prev_x = None
        while not done:
            if render:
                env.render()
                sleep(1/60)

            cur_x = preprocess(observation)
            x = cur_x + prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x

            prob = forward_pass(x, W, B, xs, hs)

            action = 4 if np.random.uniform() < prob else 3

            label = 1 if action == 4 else 0     # Fake label for calculating the grad of log(policy) wrt the output layer (log_prob)

            ds_log_prob.append(label - prob)  # If action == UP then grad = 1 - prob else grad = -prob

            observation, reward, done, info = env.step(action)

            rs.append(reward)
            ep_reward += reward

        print("Finished episode {0}. Total reward: {1}".format(ep, ep_reward))
        ep += 1
        batch_reward += ep_reward

        #  We stack all the data from the episode, but this transposes all the vectors so the backprop math is all transposed now
        xs = np.vstack(xs)
        hs = np.vstack(hs)
        ds_log_policy = np.vstack(ds_log_prob)
        rs = np.vstack(rs)

        if ep % batch_size == 0:
            L.append(batch_reward / batch_size)
            batch_reward = 0
            if plot:
                plt.plot(L)
                plt.show()

        train(W, B, xs, hs, ds_log_policy, rs)

        if ep % 100 == 0:
            with open("model.p", "wb") as f:
                pickle.dump([W, B, L], f)

        observation = env.reset()


def reshape(observation):
    return np.ravel(observation[34:-16, :, 0])


def preprocess(observation):
    x = np.ravel(observation[34:-16:2, ::2, 0])
    x[x == 144] = 0
    x[x == 109] = 0
    x[x != 0] = 1
    return x


def forward_pass(x, w, b, xs, hs):
    xs.append(x)

    h = np.dot(w[0], x) + b[0]
    h[h < 0] = 0

    hs.append(h)

    log_prob = np.dot(w[1], h) + b[1]
    return sigmoid(log_prob)


def train(w, b, xs, hs, ds_log_prob, rs):
    calculate_rewards(rs)

    rs -= np.mean(rs)
    std = np.std(rs)
    if std != 0:
        rs /= std

    ds_log_prob *= rs     # Modulate gradient with reward

    dw, db = backward_pass(w, b, xs, hs, ds_log_prob)

    for i in range(len(w)):
        w[i] += learn_rate * dw[i]

    for i in range(len(b)):
        b[i] += learn_rate * db[i]


def backward_pass(w, b, xs, hs, ds_log_prob):
    dw1 = np.dot(ds_log_prob.T, hs)
    db1 = np.sum(ds_log_prob, axis=0)

    dhs = np.dot(ds_log_prob, w[1])
    dhs[hs == 0] = 0

    dw0 = np.dot(dhs.T, xs)
    db0 = np.sum(dhs, axis=0)

    return [dw0, dw1], [db0, db1]


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
