import random
import time

start_time = time.time()
# GA parameters
population_size = 100
max_generations = 100
mutation_rate = 0.1
crossover_rate = 0.9
num_bins = 5

# problem parameters
bin_size = 10
item_sizes = [3, 4, 5, 2, 1, 7, 6, 3, 4, 5]

def create_genes():
    return [random.randint(0, num_bins-1) for i in range(len(item_sizes))]


def fitness_bins(gene):
    bins = [[] for i in range(num_bins)]
    for i, j in enumerate(gene):
        bins[j].append(item_sizes[i])
    bins_used = sum(len(bin) > 0 for bin in bins)
    unused = sum(max(0, bin_size - sum(bin)) for bin in bins)
    return bins_used + unused


def population_init(population_size):
    return [create_genes() for i in range(population_size)]


# Evolve population
def evolve():
    population = population_init(population_size)
    for generation in range(max_generations):
        # calculate fitness
        fitness_scores = [fitness_bins(gene) for gene in population]
        # Select parents
        parents = []
        for i in range(population_size):
            fitness_probs = [1 / (score + 1) for score in fitness_scores]
            parent1, parent2 = random.choices(population, weights=fitness_probs, k=2)
            parents.append((parent1, parent2))
        # mate
        offspring = []
        for parent1, parent2 in parents:
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, len(item_sizes) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                ### NDC Step goes here ###
                if fitness_bins(child1) > fitness_bins(parent1) and \
                   fitness_bins(child1) > fitness_bins(child2):
                    offspring.append(child1)
                elif fitness_bins(child2) > fitness_bins(parent2) and \
                     fitness_bins(child2) > fitness_bins(child1):
                    offspring.append(child2)
                else:
                    offspring.append(random.choice([parent1, parent2]))
                ### End of NDC ###
            else:
                offspring.append(random.choice([parent1, parent2]))
        # mutate
        for i in range(len(offspring)):
            for j in range(len(item_sizes)):
                if random.random() < mutation_rate:
                    offspring[i][j] = random.randint(0, num_bins-1)
        # select from population
        population = random.choices(offspring, k=population_size)
        print("-- in generation ",generation,":")
        best_gene = min(population, key=fitness_bins)
        best_fitness = fitness_bins(best_gene)
        print("Best solution:", best_gene)
        #print("Such that, each item is mapped to the container of the same index")
        print("Best fitness:", best_fitness)
    # Select best solution
    best_gene = min(population, key=fitness_bins)
    best_fitness = fitness_bins(best_gene)
    print("Best solution:", best_gene)
    print("Such that, each item is mapped to the container of the same index")
    print("Best fitness:", best_fitness)


if __name__ == "__main__":


    evolve()