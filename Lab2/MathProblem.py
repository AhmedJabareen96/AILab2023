import random
import math

N = 100  # population size
R1 = 3  # radius of circle 1
R2 = 2  # radius for cycle 2
k = 3  # tournament size
mu = 50  # number of selected individuals
sigma = 0.1  # standard deviation of mutation
max_generations = 1000  # maximum number of generations


def f(x, y):
    # (x-5)**2 + (y-5)**2
    return x**2 + y**2


def g(x, y):
    return (x-5)**2 + (y-5)**2


def is_valid_f(x, y, radius):
    return x**2 + y**2 <= radius**2


def is_valid_g(x, y, radius):
    return (x-5)**2 + (y-5)**2 <= radius**2

# Generate an initial population of size N and radius R


def initialize_population(size, radius, func):
    population = []
    while len(population) < size:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        if func(x, y, radius):
            population.append((x, y))
    return population

# Evaluate the fitness of each individual in the population


def evaluate_population(population, func):
    fitness_scores = []
    for individual in population:
        x, y = individual
        fitness_scores.append(-func(x, y))
    return fitness_scores

# Tournament selection with size k


def tournament_selection(population, fitness_scores, k):
    selected = []
    for i in range(k):
        index = random.randint(0, len(population) - 1)
        selected.append((population[index], fitness_scores[index]))
    return max(selected, key=lambda x: x[1])[0]

# Single-point crossover


def singlePointCrossover(parent1, parent2):
    x1, y1 = parent1
    x2, y2 = parent2
    crossover_point = random.randint(0, 1)
    if crossover_point == 0:
        return (x1, y2)
    else:
        return (x2, y1)

# Gaussian mutation with standard deviation sigma


def mutation(individual, sigma):
    x, y = individual
    x += random.gauss(0, sigma)
    y += random.gauss(0, sigma)
    return (x, y)

# genetic (exaptation) algorithm


def exaptation():
    # Initialize populations
    population_f = initialize_population(N, R1, is_valid_f)
    population_g = initialize_population(N, R1, is_valid_g)

    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores_f = evaluate_population(population_f, f)
        fitness_scores_g = evaluate_population(population_g, g)

        # Select individuals
        selected_f = [tournament_selection(
            population_f, fitness_scores_f, k) for i in range(mu)]
        selected_g = [tournament_selection(
            population_g, fitness_scores_g, k) for i in range(mu)]

        # Generate offsprings
        offspring_f = []
        offspring_g = []
        for i in range(mu // 2):
            parent1_f = random.choice(selected_f)
            parent2_f = random.choice(selected_f)
            parent1_g = random.choice(selected_g)
            parent2_g = random.choice(selected_g)

            offspring_f.append(singlePointCrossover(parent1_f, parent2_f))
            offspring_f.append(singlePointCrossover(parent2_f, parent1_f))
            offspring_g.append(singlePointCrossover(parent1_g, parent2_g))
            offspring_g.append(singlePointCrossover(parent2_g, parent1_g))

        # Mutate offsprings
        mutated_f = [mutation(individual, sigma) for individual in offspring_f]
        mutated_g = [mutation(individual, sigma) for individual in offspring_g]

        # Merge populations
        population_f = selected_f + mutated_f
        population_g = selected_g + mutated_g

        # Immigrants between two islands
        Immigrants1 = random.sample(population_f, 10)
        Immigrants2 = random.sample(population_g, 10)

        Immigrants1 = list(filter(lambda x: is_valid_g(*x, R2), Immigrants1))
        Immigrants2 = list(filter(lambda x: is_valid_f(*x, R1), Immigrants2))

        weakest_indices_f = sorted(range(len(fitness_scores_f)), key=lambda k: fitness_scores_f[k])[:len(Immigrants2)]
        weakest_indices_g = sorted(range(len(fitness_scores_g)), key=lambda k: fitness_scores_g[k])[:len(Immigrants1)]

        # Replace weakest individuals with immigrant
        for i in weakest_indices_f:
            population_f[i] = Immigrants2.pop()
            population_g[i] = Immigrants1.pop()
        # Find the best solution for each function
        best_individual_f = max(population_f, key=lambda x: f(*x))
        best_individual_g = max(population_g, key=lambda x: g(*x))
        print("------in generation ",generation," ------")
        print("Best solution for f:", best_individual_f)
        print("Best fitness for f:", f(*best_individual_f))
        print("Best solution for g:", best_individual_g)
        print("Best fitness for g:", g(*best_individual_g))
        
    # Find the best solution for each function
    best_individual_f = max(population_f, key=lambda x: f(*x))
    best_individual_g = max(population_g, key=lambda x: g(*x))
    print("------ Final result ------")
    print("Best solution for f:", best_individual_f)
    print("Best fitness for f:", f(*best_individual_f))
    print("Best solution for g:", best_individual_g)
    print("Best fitness for g:", g(*best_individual_g))


if __name__ == "__main__":

    exaptation()
