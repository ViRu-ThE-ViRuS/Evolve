'''
Example program, demonstrating use of evolve
'''

import evolve

ENV_NAME = 'LunarLander-v2'


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

    input_dim = env.observation_space.shape
    output_dim = env.action_space.n

    model = Sequential()
    model.add(Dense(input_shape=input_dim,
                    units=input_dim[0]*2,
                    activation='relu'))
    model.add(Dense(units=input_dim[0]*4,
                    activation='relu'))
    model.add(Dense(units=output_dim*8,
                    activation='relu'))
    model.add(Dense(units=output_dim*4,
                    activation='relu'))
    model.add(Dense(units=output_dim*2,
                    activation='relu'))
    model.add(Dense(units=output_dim,
                    activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


ES = evolve.EvolutionaryStrategy(
    'demo', model_function, env_function, population_size=20)

EVOLUTION_TRACK = []
for _ in range(10):
    candidate, evolution_average = ES.evolve_step()
    EVOLUTION_TRACK.append(evolution_average)

ES.performance(candidate)
