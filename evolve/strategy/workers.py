'''
Workers to run in processes
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


def _evaluate(args):
    '''
    Evaluate a given candidate

    Arguments:
        args -- [parents, model_file, env_function,
            test_episodes]

    Returns:
        [[parental_index, average rewards over test_episodes] for each parent]
    '''

    parents, model_file, env_function, test_episodes = args
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K

    model = load_model(model_file)
    env = env_function()

    parental_rewards = []
    for index, parent in enumerate(parents):
        model.set_weights(parent)

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
        parental_rewards.append([index, np.average(rewards)])

    parental_rewards = np.array(parental_rewards)
    parental_rewards = parental_rewards[parental_rewards[:, 1].argsort(
        axis=0)][::-1]

    K.clear_session()
    return parental_rewards


def _mutate(progeny, args):
    '''
        Mutates the given progeny

        Arguments:
            progeny -- progeny to mutate
            args -- [n_layers, mutation_rate, mutation]

        Returns:
            the mutated progeny
    '''

    n_layers, mutation_rate, mutation = args

    for layer in np.random.choice(
            np.arange(n_layers),
            int(np.random.sample() * (n_layers))):
        original_shape = np.array(progeny[layer]).shape
        flat = np.reshape(progeny[layer], -1)

        padded_layer = False
        if len(original_shape) == 1:
            original_shape += (1, )
            padded_layer = True

        layer_mutations = int(mutation_rate *
                              np.multiply(*original_shape))
        while layer_mutations != 0:
            chromosome_index = np.random.randint(len(flat))
            chromosome = flat[chromosome_index]

            low = chromosome - mutation * chromosome
            high = chromosome + mutation * chromosome
            flat[chromosome_index] = (
                high - low) * np.random.sample() + low
            layer_mutations -= 1

        if padded_layer:
            original_shape = original_shape[:-1]

        progeny[layer] = flat.reshape(original_shape)

    return progeny


def _crossover(parents, args):
    '''
    Crossovers the given parents to generate a progeny

    Arguments:
        parents -- [parent1, parent2] to cross over
        args -- [variable_crossed_progeny, model_file, env_function, test_episodes]

    Returns:
        the crossed progeny
    '''

    variable_crossed_progeny, model_file, env_function, test_episodes = args

    if not variable_crossed_progeny:
        crossover_rate = 0.50
        crossover_p = [crossover_rate, 1 - crossover_rate]
    else:
        evaluate_args = [parents, model_file, env_function, test_episodes]
        scores = _evaluate(evaluate_args)
        if scores[0, 0] == 0:
            p_left, p_right = scores[0, 1], scores[1, 1]
        else:
            p_left, p_right = scores[1, 1], scores[0, 1]

        total = p_left + p_right
        p_left, p_right = p_left/total, p_right/total

        crossover_p = [p_left, p_right]

    left, right = parents[0], parents[1]

    progeny = []
    for layer, _ in enumerate(left):
        original_shape = np.array(left[layer]).shape
        left_flat, right_flat = np.reshape(
            left[layer], -1), np.reshape(right[layer], -1)

        progeny_layer = []
        for index, _ in enumerate(left_flat):
            chance = np.random.choice(2, p=crossover_p)

            if chance:
                progeny_layer.append(right_flat[index])
            else:
                progeny_layer.append(left_flat[index])

        progeny.append(np.reshape(progeny_layer, original_shape))

    return progeny


def breed_worker(args):
    '''
        Breed within the given populatino to generate progeny.
        Involves crossing, mutating.

        Arguments:
            args -- [parents, n_layers, mutation_rate, mutation,
                     variable_crossed_progeny, model_file, env_function,
                     test_episodes]

        Returns:
            the bred (cross, mutated) progeny
    '''

    parents, n_layers, mutation_rate, mutation, variable_crossed_progeny, \
        model_file, env_function, test_episodes = args

    mutation_args = [n_layers, mutation_rate, mutation]
    crossover_args = [variable_crossed_progeny,
                      model_file, env_function, test_episodes]

    return _mutate(_crossover(parents, crossover_args), mutation_args)
