import random
import numpy as np
from sorting_network import SortingNetwork

class GeneticAlgorithm:
    def __init__(self, num_elements, population_size, generations):
        self.num_elements = num_elements
        self.population_size = population_size
        self.generations = generations
        self.population = []

    def generate_initial_population(self):
        for _ in range(self.population_size):
            sorting_network = SortingNetwork(self.num_elements)
            sorting_network.generate_random()
            self.population.append(sorting_network)

    def evaluate_fitness(self, input_vector):
        fitness_scores = []
        for sorting_network in self.population:
            sorted_vector = sorting_network.sort(input_vector.copy())
            fitness_scores.append(np.sum(sorted_vector == np.sort(input_vector)) / self.num_elements)
        return fitness_scores

    def selection(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        selected_population = sorted_population[:self.population_size // 2]
        return selected_population

    def crossover(self, selected_population):
        new_population = selected_population.copy()
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child = SortingNetwork(self.num_elements)
            child.comparators = np.zeros((child.num_comparators, 2), dtype=int)
            for i in range(child.num_comparators):
                parent = random.choice([parent1, parent2])
                child.comparators[i] = parent.comparators[i].copy()
            new_population.append(child)
        return new_population

    def mutation(self, population):
        for sorting_network in population:
            for i in range(sorting_network.num_comparators):
                if random.random() < 0.1:  # Mutation probability
                    sorting_network.comparators[i] = np.random.randint(0, sorting_network.num_elements, size=2)
        return population

    def evolve(self, input_vector):
        self.generate_initial_population()
        best_fitness_scores = []
        for generation in range(self.generations):
            fitness_scores = self.evaluate_fitness(input_vector)
            best_fitness_scores.append(max(fitness_scores))
            selected_population = self.selection(fitness_scores)
            new_population = self.crossover(selected_population)
            mutated_population = self.mutation(new_population)
            self.population = mutated_population
        return best_fitness_scores
