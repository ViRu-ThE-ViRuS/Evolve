'''
Tests for evolve/generate
'''

from evolve.generate import _create_model


def test__create_model():
    '''
    Test for _create_model
    '''

    import os

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

    args = ['demo', demo_model_function, demo_env_function]
    result = _create_model(*args)

    # returns filenames
    assert len(result) == 2

    # file exist
    for elem in result:
        assert isinstance(elem, str)
        assert os.path.exists(elem)

        # remove files
        os.remove(elem)
