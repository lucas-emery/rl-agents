import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import sleep


H = 50
D = 80 * 80
learn_rate = 1e-3
discount_factor = 0.99
batch_size = 10
decay_rate = 0.9

resume = True
render = False
plot = True


def main():
    env = gym.make('Pong-v0')
    observation = env.reset()

    if resume:
        with open("model.p", "rb") as f:
            model, L, data = pickle.load(f)
            grad_var_est = data["grad_var_est"]
    else:
        w1 = np.random.rand(H, D) / math.sqrt(D)    # Xavier init
        w2 = (np.random.rand(1, H) - 0.5) / math.sqrt(H)    # Deducted Uniform(0, 1) mean to eliminate tendency to go UP
                                                            # This is possible because negative values have no negative
        b1 = np.random.rand(H)                              # effects in this layer (sigmoid activation)
        b2 = np.zeros(1)

        model = [{"w": w1, "b": b1}, {"w": w2, "b": b2}]
        L = []
        grad_var_est = [{k: np.zeros_like(v) for k, v in layer.items()} for layer in model]

    ep = 0
    batch_reward = 0
    grad_buffer = [{k: np.zeros_like(v) for k, v in layer.items()} for layer in model]
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

            prob = forward_pass(x, model, xs, hs)

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

        grad = calculate_gradients(model, xs, hs, ds_log_policy, rs)

        for layer in range(len(model)):
            for k in model[layer].keys():
                grad_buffer[layer][k] += grad[layer][k]

        if ep % batch_size == 0:
            L.append(batch_reward / batch_size)
            batch_reward = 0

            train(model, grad_buffer, grad_var_est)
            grad_buffer = [{k: np.zeros_like(v) for k, v in layer.items()} for layer in model]

            if plot:
                plt.plot(L)
                plt.show()

        if ep % 100 == 0:
            with open("model.p", "wb") as f:
                pickle.dump([model, L, {"grad_var_est": grad_var_est}], f)

        observation = env.reset()


def reshape(observation):
    return np.ravel(observation[34:-16, :, 0])


def preprocess(observation):
    x = np.ravel(observation[34:-16:2, ::2, 0])
    x[x == 144] = 0
    x[x == 109] = 0
    x[x != 0] = 1
    return x


def forward_pass(x, model, xs, hs):
    xs.append(x)

    h = np.dot(model[0]["w"], x) + model[0]["b"]
    h[h < 0] = 0

    hs.append(h)

    log_prob = np.dot(model[1]["w"], h) + model[1]["b"]
    return sigmoid(log_prob)


def calculate_gradients(model, xs, hs, ds_log_prob, rs):
    calculate_rewards(rs)

    rs -= np.mean(rs)
    std = np.std(rs)
    if std != 0:
        rs /= std

    ds_log_prob *= rs     # Modulate gradient with reward

    return backward_pass(model, xs, hs, ds_log_prob)


def train(model, grad, grad_var_est):
    for layer in range(len(model)):
        for k in model[layer].keys():
            grad_var_est[layer][k] = decay_rate * grad_var_est[layer][k] + (1 - decay_rate) * (grad[layer][k] * grad[layer][k])
            model[layer][k] += learn_rate * grad[layer][k] / (np.sqrt(grad_var_est[layer][k]) + 1e-8)


def backward_pass(model, xs, hs, ds_log_prob):
    dw1 = np.dot(ds_log_prob.T, hs)
    db1 = np.sum(ds_log_prob, axis=0)

    dhs = np.dot(ds_log_prob, model[1]["w"])
    dhs[hs == 0] = 0

    dw0 = np.dot(dhs.T, xs)
    db0 = np.sum(dhs, axis=0)

    return [{"w": dw0, "b": db0}, {"w": dw1, "b": db1}]


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
