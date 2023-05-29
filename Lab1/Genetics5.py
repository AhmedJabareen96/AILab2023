import math
import random
import time
import copy
import Gene
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Nqueens import NQueens
import BinPackaging
import calculations

# Define the fitness function
def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score

def bulls_hit(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if(individual[i] == target[i]): # if guessed the correct char at the correct place
            score += 3
        else:
            if not individual[i] in target: # if the char is not in the string then give a penalty
                score -= 1
    return score

# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations,cross_type):
    start_time = time.time()
    # Initialize the population with random individuals
    overall_time = time.time()
    population = init_population(pop_size, [], num_genes,fitness_func)
    avgs = []
    # Evolve the population for a fixed number of generations
    _age = 0
    for individual in population:
        individual.age = _age
    for generation in range(max_generations):
        _age+=1
        for individual in population:
            individual.age = _age
        # Evaluate the fitness of each Gene
        iteration_time = time.time()
        fitnesses = [individual.fitness for individual in population]
        strs = [individual.str for individual in population]
        avg=sum(fitnesses)/len(fitnesses)
        all =0
        for i in range(len(fitnesses)):
            all +=  abs(fitnesses[i] -avg )**2
        standard_dev = all/len(fitnesses)
        standard_dev =math.sqrt(standard_dev)
        best=max(population, key=lambda individual: individual.fitness).str
        print("in generation " ,generation," the average is: ",avg," std is: ",standard_dev ,"best gene:",
              ''.join(best ) )
        calculations.selection_pressure(fitnesses,int(pop_size*0.1),pop_size)
        calculations.genetic_diversification(strs)
        if best == list("Hello, world!"):
            print("\n global optimum found in ",generation," generations")
            break
        # Select the best individuals for reproduction
        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]
        avgs.append(avg)
        # Generate new individuals by applying crossover and mutation operators
        offspring = []
        while len(offspring) < pop_size - elite_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = parent1.str
            if cross_type=="SINGLE":
                pos = random.randint(0,pop_size-1)
                child= parent1.str[0: pos]+parent2.str[pos:]
            if cross_type=="TWO":
                pos1 = random.randint(0,pop_size/2-1)
                pos2 = random.randint(pop_size/2,pop_size-1)
                child = parent1.str[0:pos1]+parent2.str[pos1:pos2]+parent1.str[pos2:]
            if(cross_type=="UNIFORM"):
                child = [parent1.str[i] if random.random() < 0.5 else parent2.str[i] for i in range(num_genes)]
            # perform mutation
            if random.uniform(0, 1) < 0.3 :
                child = mutate(child)
            new_offspring = Gene.Gene(child,fitness_func(child),generation)
            new_offspring.age = 0
            offspring.append(new_offspring)
        population = elites + offspring
        clock_ticks = time.time() - iteration_time
        print("Clock ticks time: ", clock_ticks)
    overallclock_ticks = time.time() - overall_time
    # Find the Gene with the highest fitness
    best_individual = max(population, key=lambda individual: individual.fitness)
    best_fitness = best_individual.fitness
    if best_fitness == num_genes:
        print("time to get global optimum is ",overallclock_ticks)
    else:
        print("time to get local optimum is :",overallclock_ticks)
    if generation == 99:
        generation+=1
    histogram(avgs,generation)
    return best_individual.str, best_fitness

def mutate(individual):
    tsize = len(list("Hello, World!"))
    ipos = random.randint(0, tsize - 1)
    delta = random.randint(0, 90) + 32
    individual[ipos] = chr(((ord(individual[ipos]) + delta) % 122))
    return individual

def init_population(pop_size, population, num_genes,fitness_func) :
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        gene = Gene.Gene(individual,fitness_func(individual),0)
        population.append( gene )
    return population

def histogram(fitness,pop_size):
    x = [0] * pop_size
    for i in range(pop_size):
        x[i] = i
    plt.scatter(x, fitness)
    plt.title('The distribution of the genes fitnesses')
    plt.xlabel('generation')
    plt.ylabel('Fitness Average')
    plt.show()

def nqueens(popsize,maxiter,elite_rate,mutation_rate):
    pmx = int(input("CrossOver type : PMX OR CX :\n1 --> PMX  |  0 --> CX\n"))
    mutateType = int(input("MUTATION TYPE :\n1 --> Shuffle  |  0 --> inversion\n"))
    pmx = pmx == 1
    numQueens = int(input("NQueens Number :\n"))
    nq = NQueens(numQueens, popsize, maxiter, elite_rate, mutation_rate, pmx, mutateType,0)
    nq.run()
# Run the genetic algorithm and print the result
if __name__ == "__main__":

    while True:
        problem = int(input("choose :\n0 --> Genetic  |  1 --> Nqueens | 2 -->BinPacking | 3--> exit\n"))
        if problem == 0:
            best_individual, best_fitness = genetic_algorithm(pop_size=200, num_genes=13, fitness_func=bulls_hit, max_generations=100,cross_type='UNIFORM')
            print("Best Gene:", ''.join(best_individual))
            print("Best fitness:", best_fitness)
        if problem == 1:
            nqueens(100,100,0.1,0.3)
        if problem ==2:
            BinPackaging.evolve()
        if problem == 3:
            break


