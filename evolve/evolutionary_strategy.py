from multiprocessing.pool import Pool
import multiprocessing as mp
import pickle
import os
import logging
import numpy as np

from .generate import create_model

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_worder(args):
    weights_file = args[0]
    with open(weights_file, 'rb') as file:
        weights = pickle.load(file)

    return weights


def evaluate_worker(args):
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


class EvolutionaryStrategy():
    evolution = 0
    _previous_scores = -1

    def __init__(self, name, model_function, env_function,
                 population_size=50,
                 mutation=3.0, mutation_rate=0.80, mutation_decay=True,
                 mutation_decay_rate=0.99,
                 variable_crossed_progeny=True,
                 selection_cutoff_decay=True,
                 selection_cutoff_decay_rate=0.95, selection_cutoff=0.20,
                 test_episodes=5):
        self.name = name
        self.env_function = env_function

        self.model_file, self.weights_file = create_model(
            self.name, self.env_function, model_function)

        self.population_size = population_size

        self.variable_crossed_progeny = variable_crossed_progeny
        self.test_episodes = test_episodes

        self.mutation_rate = mutation_rate
        self.mutation_decay_rate = mutation_decay_rate if mutation_decay else 1
        self.mutation = mutation

        self.selection_cutoff_decay_rate = selection_cutoff_decay_rate \
            if selection_cutoff_decay else 1
        self.selection_cutoff = selection_cutoff

        self.pool = Pool(processes=mp.cpu_count())

        args = [[self.weights_file]]
        results = []
        for result in self.pool.imap(load_worder, args):
            results.append(result)
        weights = results[0]

        self.population = []
        for _ in range(self.population_size):
            candidate = []
            for weight in weights:
                candidate.append(np.random.randn(*weight.shape))
            self.population.append(candidate)
        self.n_layers = len(self.population[0])

        os.remove(self.weights_file)

    def evolve_step(self, return_population=False):
        self.evolution += 1
        print('EVOLUTION {}'.format(self.evolution))

        print('\tselecting from population...')
        if self.evolution - 1:
            self.selection_cutoff *= self.selection_cutoff_decay_rate

        n_selected = int(self.population_size * self.selection_cutoff)

        if not isinstance(self._previous_scores, int):
            scores = self._previous_scores
        else:
            scores = self._evaluate(self.population)

        lucky_factor = 0.20
        top_selection, bottom_selection = int(
            n_selected * (1 - lucky_factor)), int(n_selected * lucky_factor)
        scores, scrap = scores[:n_selected], scores[n_selected:]
        scores = scores[np.random.choice(np.arange(n_selected), top_selection)]
        scrap = scrap[np.random.choice(
            np.arange(self.population_size - n_selected), bottom_selection)]
        selected_candidates = np.vstack((scores, scrap))

        selected_population = []
        for index in np.array(selected_candidates[:, 0], dtype=np.int32):
            selected_population.append(self.population[index])

        print('\tbreeding from selected population...')
        n_bred = self.population_size
        progeny = self._breed(selected_population, n_bred)

        print('\tevaluating progeny...')
        self.population = progeny
        generation_evaluation = self._evaluate(self.population)
        best_performance = generation_evaluation[0, 1]
        average_performance = np.average(generation_evaluation[:, 1])

        print('evolution {}: top_generation_performance = {}, '
              'average_generation_performance = {}'.format(
                  self.evolution, best_performance, average_performance))

        self._previous_scores = generation_evaluation

        if not return_population:
            return self.population[int(generation_evaluation[0, 0])], \
                average_performance
        else:
            return self.population, average_performance

    def _evaluate(self, population, test_episodes=None):
        if not test_episodes:
            test_episodes = self.test_episodes

        args = []
        for index, candidate in enumerate(population):
            args.append([candidate, self.model_file,
                         self.env_function, test_episodes, index])

        scores = []
        for result in self.pool.imap_unordered(evaluate_worker, args):
            scores.append(result)

        scores = np.array(scores)
        scores = scores[scores[:, 1].argsort(axis=0)][::-1]

        return scores

    def _breed(self, population, progeny_to_generate):
        bred = []
        for _ in range(progeny_to_generate):
            left, right = np.random.choice(len(population), 2)
            parents = [population[left], population[right]]
            progeny = self._mutate(self._crossover(parents))
            bred.append(progeny)

        return np.array(bred)

    def _mutate(self, progeny):
        if self.evolution - 1:
            self.mutation_rate *= self.mutation_decay_rate
            self.mutation *= self.mutation_decay_rate

        for layer in np.random.choice(
                np.arange(self.n_layers),
                int(np.random.sample() * (self.n_layers))):
            original_shape = np.array(progeny[layer]).shape
            flat = np.reshape(progeny[layer], -1)

            layer_mutations = int(self.mutation_rate *
                                  np.multiply(*original_shape))
            while layer_mutations != 0:
                chromosome_index = np.random.randint(len(flat))
                chromosome = flat[chromosome_index]

                low = chromosome - self.mutation * chromosome
                high = chromosome + self.mutation * chromosome
                flat[chromosome_index] = (
                    high - low) * np.random.sample() + low
                layer_mutations -= 1

            progeny[layer] = flat.reshape(original_shape)

        return progeny

    def _crossover(self, parents):
        if not self.variable_crossed_progeny:
            crossover_rate = 0.50
            crossover_p = [crossover_rate, 1 - crossover_rate]
        else:
            scores = self._evaluate(parents)
            if scores[0, 0] == 0:
                p_left, p_right = scores[0, 1], scores[1, 1]
            else:
                p_left, p_right = scores[1, 1], scores[0, 1]

            crossover_p = [p_left / (p_left + p_right),
                           p_right / (p_left + p_right)]

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

    def performance(self, candidate, test_episodes=50, get_rewards=False):
        args = [[candidate, self.model_file, self.env_function,
                 test_episodes]]

        rewards = []
        for result in self.pool.imap_unordered(
                performance_worker, args):
            rewards.append(result)

        print('\nmodel_average_performance over {} episodes = {}'
              .format(test_episodes, np.average(rewards)))
        print('model_peak_performance = {}, model_min_performance = {}'
              .format(np.max(rewards), np.min(rewards)))

        if get_rewards:
            return rewards
