import random
import numpy as np
import matplotlib.pyplot as plt
from sorting_network import SortingNetwork

MUTATION_RATE = 0.1  # try tuning this

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
    selected_population = [population[index] for index in selected_indices]
    return selected_population

class Coevolution:
    def __init__(self, num_elements, population_size, generations):
        self.num_elements = num_elements
        self.population_size = population_size
        self.generations = generations
        self.network_population = []
        self.vector_population = []

    def generate_initial_population(self):
        for _ in range(self.population_size):
            network = SortingNetwork(self.num_elements)
            vector = np.random.randint(1, 17, size=self.num_elements)
            self.network_population.append(network)
            self.vector_population.append(vector)

    def evaluate_fitness(self):
        network_fitness_scores = []
        vector_fitness_scores = []
        comparator_counts = []
        for network, vector in zip(self.network_population, self.vector_population):
            sorted_vector = network.sort(vector)
            network_fitness = sum(sorted_vector == np.sort(vector))
            vector_fitness = self.population_size - sum(sorted_vector == np.sort(vector))
            network_fitness_scores.append(network_fitness)
            vector_fitness_scores.append(vector_fitness)
            comparator_counts.append(network.num_comparators)
        return network_fitness_scores, vector_fitness_scores, comparator_counts

    def selection(self, network_fitness_scores, vector_fitness_scores):
        # Select parents based on fitness scores and comparator counts
        combined_scores = [network_fitness_scores[i] + vector_fitness_scores[i] for i in range(self.population_size)]
        selected_network_population = roulette_wheel_selection(self.network_population, combined_scores)
        selected_vector_population = roulette_wheel_selection(self.vector_population, combined_scores)
        return selected_network_population, selected_vector_population

    def crossover(self, selected_network_population, selected_vector_population):
        new_network_population = []
        new_vector_population = []
        for _ in range(self.population_size):
            parent1_network = random.choice(selected_network_population)
            parent2_network = random.choice(selected_network_population)
            child_network = SortingNetwork(self.num_elements)
            child_network.comparators = np.where(np.random.rand(*child_network.comparators.shape) < 0.5,
                                                 parent1_network.comparators,
                                                 parent2_network.comparators)
            if random.random() < MUTATION_RATE:  # Mutation probability
                for i in range(child_network.num_comparators):
                    if random.random() < 0.1:  # Mutation probability
                        child_network.comparators[i] = np.random.randint(0, self.num_elements, size=2)
            new_network_population.append(child_network)
            new_vector_population.append(random.choice(selected_vector_population))
        return new_network_population, new_vector_population

    def evolve(self):
        self.generate_initial_population()
        best_network_fitness_scores = []
        best_vector_fitness_scores = []
        comparator_counts = []
        for generation in range(self.generations):
            network_fitness_scores, vector_fitness_scores, comparators = self.evaluate_fitness()
            best_network_fitness_scores.append(max(network_fitness_scores))
            best_vector_fitness_scores.append(max(vector_fitness_scores))
            comparator_counts.append(min(comparators))
            selected_network_population, selected_vector_population = self.selection(network_fitness_scores, vector_fitness_scores)
            new_network_population, new_vector_population = self.crossover(selected_network_population, selected_vector_population)
            self.network_population, self.vector_population = new_network_population, new_vector_population
        return best_network_fitness_scores, best_vector_fitness_scores, comparator_counts