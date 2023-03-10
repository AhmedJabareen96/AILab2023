import math
import random
import  time


# Define the fitness function
def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score

# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations):
    # Initialize the population with random individuals
    overall_time = time.time()
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    # Evolve the population for a fixed number of generations
    for generation in range(max_generations):
        # Evaluate the fitness of each individual
        iteration_time = time.time()
        fitnesses = [fitness_func(individual) for individual in population]
        avg=sum(fitnesses)/len(fitnesses)
        standard_dev=0
        all =0
        for i in range(len(fitnesses)):
            all +=  abs(fitnesses[i] -avg )**2
        standard_dev = all/len(fitnesses)
        standard_dev =math.sqrt(standard_dev)
        print("in generation " ,generation," the average is: ",avg," std is: ",standard_dev )
        # Select the best individuals for reproduction
        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes)]
            offspring.append(child)
        population = elites + offspring
        clock_ticks = time.time() - iteration_time
        print("Clock ticks time: ", clock_ticks)
    overallclock_ticks = time.time() - overall_time
    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)
    if best_fitness == num_genes:
        print("time to get global optimum is ",overallclock_ticks)
    else:
        print("time to get local optimum is :",overallclock_ticks)
    return best_individual, best_fitness

# Run the genetic algorithm and print the result
best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=13, fitness_func=fitness, max_generations=100)
print("Best individual:", ''.join(best_individual))
print("Best fitness:", best_fitness)
