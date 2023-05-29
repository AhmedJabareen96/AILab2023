import random

# Roulette Wheel Selection Algorithm
def roulette_wheel_selection(fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    r = random.random()
    for i in range(len(cumulative_probabilities)):
        if r < cumulative_probabilities[i]:
            return i

# RWS with scaled fitness values
def roulette_wheel_selection_with_scaling(fitness_values):
    """Performs roulette wheel selection with scaling on a list of fitness values."""
    scaled_fitness_values = [fitness**2 for fitness in fitness_values]
    total_scaled_fitness = sum(scaled_fitness_values)
    probabilities = [fitness / total_scaled_fitness for fitness in scaled_fitness_values]
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    r = random.random()
    for i in range(len(cumulative_probabilities)):
        if r < cumulative_probabilities[i]:
            return i


# SUS algorithm
def stochastic_universal_sampling(fitness_values, num_samples):
    total_fitness = sum(fitness_values)
    fitness_per_sample = total_fitness / num_samples
    start = random.uniform(0, fitness_per_sample)
    pointers = [start + i * fitness_per_sample for i in range(num_samples)]
    selected_indices = []
    for pointer in pointers:
        i = 0
        while sum(fitness_values[:i+1]) < pointer:
            i += 1
        selected_indices.append(i)
    return selected_indices

import random

# SUS with scaling
def stochastic_universal_sampling_with_scaling(fitness_values, num_samples):
    scaled_fitness_values = [fitness**2 for fitness in fitness_values]
    total_scaled_fitness = sum(scaled_fitness_values)
    fitness_per_sample = total_scaled_fitness / num_samples
    start = random.uniform(0, fitness_per_sample)
    pointers = [start + i * fitness_per_sample for i in range(num_samples)]
    selected_indices = []
    for pointer in pointers:
        i = 0
        cumulative_fitness = scaled_fitness_values[i]
        while cumulative_fitness < pointer:
            i += 1
            cumulative_fitness += scaled_fitness_values[i]
        selected_indices.append(i)
    return selected_indices


def tournament_selection(population, k):
    tournament = random.sample(population, k)
    winner = max(tournament, key=lambda x: x.fitness)
    return winner


def tournament_selection_with_rank(population, k):
    ranked_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    for i in range(len(ranked_population)):
        ranked_population[i].rank = i+1
    tournament = random.sample(ranked_population, tournament_size)
    winner = max(tournament, key=lambda x: x.rank)
    return winner
    