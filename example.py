import evolve

env_name = 'CartPole-v1'


def env_function():
    import gym
    return gym.make(env_name)


def model_function(env):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(input_shape=env.observation_space.shape,
                    units=16, activation='linear', use_bias=False))
    model.add(Dense(units=8, activation='linear', use_bias=False))
    model.add(Dense(units=env.action_space.n,
                    activation='linear', use_bias=False))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


es = evolve.EvolutionaryStrategy('demo', model_function, env_function)

evolution_track = []
for _ in range(5):
    candidate, evolution_average = es.evolve_step()
    evolution_track.append(evolution_average)

es.performance(candidate)
