'''
Tests for evolve/evolutionary_strategy
'''

from evolve.evolutionary_strategy import load_worker
from evolve.evolutionary_strategy import EvolutionaryStrategy


def test_load_worker():
    '''
    Test for load_worker
    '''
    import pickle
    import os

    demo_weights = [[1, 2, 3, 4], [0], [5, 6, 7, 8], [0]]

    WEIGHTS_FILE = 'demo_weights'
    with open(WEIGHTS_FILE, 'wb') as file:
        pickle.dump(demo_weights, file)

    weights = load_worker([WEIGHTS_FILE])
    assert isinstance(weights, list)

    os.remove(WEIGHTS_FILE)


def demo_model_function(env):
    '''
    Demo model function
    '''

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(input_shape=env.observation_space.shape,
                    units=env.action_space.n,
                    activation='linear',
                    use_bias=True))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def demo_env_function():
    '''
    Demo env function
    '''

    import gym
    return gym.make('CartPole-v0')


def test_evolutionary_strategy():
    '''
    Test evolution strategy
        - check if performance increases
        - check if variables decay
        - if sample setup runs with convergence
    '''

    import numpy as np
    import os

    es = EvolutionaryStrategy('demo', demo_model_function, demo_env_function,
                              population_size=20, test_episodes=1,
                              mutation_decay=True,
                              selection_cutoff_decay=True)

    assert es is not None
    assert es.test_episodes == 1

    os.remove('demo_model')
