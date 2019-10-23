from multiprocessing import Process, Queue


def _create_model_worker(_q, name, env_function, model_function):
    model_file = name + '_model'
    weights_file = name + '_weights'

    import tensorflow.keras.backend as K
    import pickle

    model = model_function(env_function())
    model.save(model_file)
    print('Model saved in file = {}'.format(model_file))

    weights_structure = model.get_weights()
    pickle.dump(weights_structure, open(weights_file, "wb"))
    print('Weights saved in file = {}'.format(weights_file))

    K.clear_session()
    _q.put([model_file, weights_file])


def _create_model(name, model_function, env_function):
    _q = Queue()
    worker = Process(
        target=_create_model_worker,
        args=[_q, name, env_function, model_function])
    worker.start()
    worker.join()

    return _q.get()
