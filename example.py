'''
Example program, demonstrating use of evolve
'''

import evolve

ENV_NAME = 'CartPole-v1'


def env_function():
    '''
    Returns a new environment
    '''

    import gym
    return gym.make(ENV_NAME)


def model_function(env):
    '''
    Returns a keras model
    '''

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(input_shape=env.observation_space.shape,
                    units=16,
                    activation='linear',
                    use_bias=True))
    model.add(Dense(units=8,
                    activation='linear',
                    use_bias=True))
    model.add(Dense(units=env.action_space.n,
                    activation='linear',
                    use_bias=True))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


ES = evolve.EvolutionaryStrategy('demo', model_function, env_function,
                                 population_size=100)

EVOLUTION_TRACK = []
for _ in range(15):
    candidate, evolution_average = ES.evolve_step()
    EVOLUTION_TRACK.append(evolution_average)

ES.performance(candidate)
