'''
Workers to run in parallel
'''

import pickle
import numpy as np


def load_worker(args):
    '''
    Load the model weights

    Arguments:
        args -- [weights_file]

    Returns:
        The weights list from the weights_file
    '''

    weights_file = args[0]
    with open(weights_file, 'rb') as file:
        weights = pickle.load(file)

    return weights


def evaluate_worker(args):
    '''
    Evaluate a given candidate

    Arguments:
        args -- [candidate, model_file, env_function,
            test_episodes, population_index]

    Returns:
        [population_index, average rewards over test_episodes]
    '''

    candidate, model_file, env_function, test_episodes, index = args
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K

    model = load_model(model_file)
    model.set_weights(candidate)
    env = env_function()

    rewards = []
    for _ in range(test_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(model.predict(
                np.expand_dims(state, axis=0)))
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    K.clear_session()
    return [index, np.average(rewards)]


def performance_worker(args):
    '''
    Evaluate the performance of the given candidate

    Arguments:
        args -- [candidate, model_file, env_function, test_episodes]

    Returns:
        rewards of the candidate over test_episodes
    '''

    candidate, model_file, env_function, test_episodes = args

    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K

    model = load_model(model_file)
    model.set_weights(candidate)
    env = env_function()

    rewards = []
    for _ in range(test_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(model.predict(
                np.expand_dims(state, axis=0)))
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    K.clear_session()
    return rewards
