import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import sleep

learn_rate = None
discount_factor = None
batch_size = None
var_decay = None
weight_decay = None
division_epsilon = None
var_bias = None

resume = True
render = False
plot = True


def main():
    global learn_rate
    global discount_factor
    global batch_size
    global var_decay
    global weight_decay
    global division_epsilon
    global var_bias

    env = gym.make('Pong-v0')
    observation = env.reset()

    if resume:
        with open("model.p", "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            history = data["history"]
            grad_var_est = data["grad_var_est"]
            episode = data["episode"]
            H = data["H"]
            D = data["D"]
            learn_rate = data["learn_rate"]
            discount_factor = data["discount_factor"]
            batch_size = data["batch_size"]
            var_decay = data["var_decay"]
            weight_decay = data["weight_decay"]
            division_epsilon = data["division_epsilon"]

            var_bias = var_decay ** episode

            ema = [-21.0]
            for i in range(len(history)):
                ema.append(0.95 * ema[i] + 0.05 * history[i])
    else:
        H = 100
        D = 80 * 80
        learn_rate = 1e-3
        discount_factor = 0.99
        batch_size = 10
        var_decay = 0.9
        weight_decay = 1e-3
        division_epsilon = 1e-8
        episode = 0

        w1 = np.random.randn(H, D) / math.sqrt(D)       # Xavier init
        w2 = np.random.randn(1, H) / math.sqrt(H)

        b1 = np.zeros(H)
        b2 = np.zeros(1)

        model = [{"w": w1, "b": b1}, {"w": w2, "b": b2}]
        history = []
        ema = [-21.0]
        grad_var_est = [{k: np.zeros_like(v) for k, v in layer.items()} for layer in model]
        var_bias = 1

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
            x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x

            prob = forward_pass(x, model, xs, hs)

            action = 2 if np.random.uniform() < prob else 3

            label = 1 if action == 2 else 0     # Fake label for calculating the grad of log(policy) wrt the output layer (log_prob)

            ds_log_prob.append(label - prob)  # If action == UP then grad = 1 - prob else grad = -prob

            observation, reward, done, info = env.step(action)

            rs.append(reward)
            ep_reward += reward

        episode += 1
        print("Finished episode {0}. Total reward: {1}".format(episode, ep_reward))
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

        if episode % batch_size == 0:
            history.append(batch_reward / batch_size)
            batch_reward = 0

            train(model, grad_buffer, grad_var_est)
            grad_buffer = [{k: np.zeros_like(v) for k, v in layer.items()} for layer in model]

            if plot:
                ema.append(0.95 * ema[-1] + 0.05 * history[-1])
                plt.plot(history)
                plt.plot(ema[1:])
                plt.show()

        if episode % 100 == 0:
            with open("model.p", "wb") as f:
                pickle.dump({"model": model, "history": history, "H": H, "D": D, "episode": episode, "learn_rate": learn_rate,
                             "grad_var_est": grad_var_est, "discount_factor": discount_factor, "var_decay": var_decay,
                             "batch_size": batch_size, "weight_decay": weight_decay, "division_epsilon": division_epsilon}, f)

        observation = env.reset()


def reshape(observation):
    return np.ravel(observation[34:-16, :, 0])


def preprocess(observation):
    x = np.ravel(observation[34:-16:2, ::2, 0])
    x[x == 144] = 0
    x[x == 109] = 0
    x[x != 0] = 1
    return x.astype(np.float)


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
    global var_bias
    var_bias *= var_decay
    for layer in range(len(model)):
        for k in model[layer].keys():
            grad_var_est[layer][k] = var_decay * grad_var_est[layer][k] + (1 - var_decay) * (grad[layer][k] * grad[layer][k])
            model[layer][k] += learn_rate * grad[layer][k] / (np.sqrt(grad_var_est[layer][k] / (1 - var_bias)) + division_epsilon)
            model[layer][k] *= 1 - weight_decay


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
