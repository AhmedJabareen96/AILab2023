import random
import numpy as np
from sorting_bionic import BionicSorting

class GeneticAlgorithm:
    def __init__(self, num_elements, population_size, generations):
        self.num_elements = num_elements
        self.population_size = population_size
        self.generations = generations
        self.population = []

    def generate_initial_population(self):
        for _ in range(self.population_size):
            sorting_bionic = BionicSorting(self.num_elements)
            sorting_bionic.generate_random()
            self.population.append(sorting_bionic)

    def evaluate_fitness(self, input_vector):
        fitness_scores = []
        for sorting_bionic in self.population:
            sorted_vector = sorting_bionic.sort(input_vector.copy())
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
            child = BionicSorting(self.num_elements)
            child.permutation = parent1.crossover(parent2).permutation
            new_population.append(child)
        return new_population

    def mutation(self, population):
        for sorting_bionic in population:
            if random.random() < 0.1:  # Mutation probability
                np.random.shuffle(sorting_bionic.permutation)
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
