import math
import random
import time
import copy
import Gene
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from Nqueens import NQueens
import BinPackagingSpeciatin
import calculations
import BinPackagingSpeciatin
import BinPackagingNiche
import BinPackagingNDC
import MathProblem
hypermutation_rate = 0.8
# Define the trigger condition
trigger_condition = 5 # number of generations without improvement

# Define the hypermutation period
hypermutation_period = 5 # number of generations with high mutation rate


#Encode the target string in binary form
target = ''.join(format(ord(c), '08b') for c in "Hello, world!")
num_genes = len(target)
def fitness(individual):
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score

# Define the fitness function
def fitness_binary(individual):
    #target = list("Hello, world!")
    # Convert the string to bytes
    my_bytes = ''.join(target).encode()

    # Convert the bytes to binary representation
    target_binary = (bin(int.from_bytes(my_bytes, byteorder='big')))

    ind_bytes = ''.join(individual).encode()

    # Convert the bytes to binary representation
    ind_binary = (bin(int.from_bytes(ind_bytes, byteorder='big')))
    score = 0
    for i in range(len(ind_binary)):
        if ind_binary[i] == target_binary[i]:
            score += 3
        else:
            if not ind_binary[i] in target_binary: # if the char is not in the string then give a penalty
                score -= 1
    return score

def bulls_hit(individual):
    #target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if(individual[i] == target[i]): # if guessed the correct char at the correct place
            score += 3
        else:
            if not individual[i] in target: # if the char is not in the string then give a penalty
                score -= 1
    return score

# Define the genetic algorithm
def apply_mutation(population, hypermutation_rate,pop_size):
    for i in range (pop_size):
        population[i].str=binary_mutate(list(population[i].str))

    pass


def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations,cross_type,mutationtype,mutation_rate):
    start_time = time.time()
    # Initialize the population with random individuals
    overall_time = time.time()
    population = init_population(pop_size, [], num_genes,fitness_func)
    avgs = []
    # Evolve the population for a fixed number of generations
    _age = 0
    last_best = -100
    for individual in population:
        individual.age = _age
    for generation in range(max_generations):
        _age+=1
        for individual in population:
            individual.age = _age
        # Evaluate the fitness of each Gene
        iteration_time = time.time()
        fitnesses = [individual.fitness for individual in population]
        # Check if the best fitness has improved
        if max(fitnesses) > last_best:
            last_best = fitnesses[0]
            best_solution = fitnesses[0]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        # Check if the trigger condition is met
        if mutationtype==4:
            if no_improvement_count >= trigger_condition:
                # Apply hypermutation for the specified period
                for i in range(hypermutation_period):
                    # Apply mutation with the high mutation rate
                    population = apply_mutation(population, hypermutation_rate,pop_size)

                    # # Evaluate the fitness of the mutated population
                    # sorted_fitness = evaluate_fitness(sorted_population)
                    #
                    # # Sort the mutated population by fitness
                    # sorted_population, sorted_fitness = sort_population(sorted_population, sorted_fitness)
        if mutationtype == 5:
            for i in range(pop_size):
                population[i].update_mutationrate()

        strs = [individual.str for individual in population]
        avg=sum(fitnesses)/len(fitnesses)
        all =0
        for i in range(len(fitnesses)):
            all +=  abs(fitnesses[i] -avg )**2
        standard_dev = all/len(fitnesses)
        standard_dev =math.sqrt(standard_dev)
        best=''.join(max(population, key=lambda individual: individual.fitness).str)
        decoded_best_individual = ''.join(chr(int(best[i:i + 8], 2)) for i in range(0, num_genes, 8))
        print("in generation " ,generation," the average is: ",avg," std is: ",standard_dev ,"best gene:",
              ''.join(decoded_best_individual ) )
        tsp=calculations.selection_pressure(fitnesses,int(pop_size*0.1),pop_size)
        calculations.genetic_diversification(strs)
        if best == target:
            print("\n global optimum found in ",generation," generations")
            break
        # Select the best individuals for reproduction
        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]
        avgs.append(avg)
        # Generate new individuals by applying crossover and mutation operators
        offspring = []

        if mutationtype == 2:
            mutation_rate -= 0.03
        if mutationtype == 3:
            mutation_rate*=tsp

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
            if random.uniform(0, 1) < mutation_rate :
                child = binary_mutate(child)
            new_offspring = Gene.Gene(child,fitness_func(child),generation)
            if mutationtype==5 and random.uniform(0, 1) < new_offspring.mut_rate:
                new_offspring.update_mutationrate()
                new_offspring.str=binary_mutate(list(child))
            new_offspring.age = 0
            offspring.append(new_offspring)
        if mutationtype == 3:
            mutation_rate*=tsp
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
    st=''.join(best_individual.str)
    decoded_best_individual = ''.join(chr(int(st[i:i + 8], 2)) for i in range(0, num_genes, 8))
    return decoded_best_individual, best_fitness,generation

def mutate(individual):
    tsize = len(list("Hello, World!"))
    ipos = random.randint(0, tsize - 1)
    delta = random.randint(0, 90) + 32
    individual[ipos] = chr(((ord(individual[ipos]) + delta) % 122))
    return individual

def binary_mutate(individual):
    gene_index = random.randint(0, num_genes - 1)
    individual[gene_index] = str(int(not int(individual[gene_index])))
    individual = ''.join(individual)
    return individual

def binary_crossover(individual):
    tsize = len(list("Hello, World!"))
    ipos = random.randint(0, tsize - 1)
    delta = random.randint(0, 90) + 32
    individual[ipos] = chr(((ord(individual[ipos]) + delta) % 122))
    return individual

def init_population(pop_size, population, num_genes,fitness_func) :
    for i in range(pop_size):
        individual = ''.join(str(random.randint(0, 1)) for j in range(num_genes))
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
        problem = int(input("choose :\n0 --> Genetic  |  1 --> Nqueens | 2 -->BinPackingNDC | 3--> BinPackingNieche  |  4-->BinpackingSpeciating  |  5-->Math problem  | 6-->exit\n"))
        if problem == 0:
            mt=int(input("choose mutation type:\n1 --> basic  |  2 --> Non-Unform Mutation | 3 -->Adaptive Mutation | 4--> THM | 5-->Self-Adaptive\n"))
            best_individual, best_fitness,generations = genetic_algorithm(pop_size=200, num_genes=num_genes, fitness_func=bulls_hit,
                                                                          max_generations=100,cross_type='UNIFORM',mutationtype=mt,mutation_rate=0.3)
            print("Best Gene:", ''.join(best_individual))
            print("Best fitness:", best_fitness)
            # avg_bull_hit=0
            # avg_regular=0
            # for i in range (50):
            #     best_individual, best_fitness,generations = genetic_algorithm(pop_size=100, num_genes=num_genes, fitness_func=bulls_hit, max_generations=100,cross_type='UNIFORM')
            #     avg_bull_hit+=generations
            # avg_bull_hit/=50
            # for i in range (50):
            #     best_individual, best_fitness,generations = genetic_algorithm(pop_size=100, num_genes=num_genes, fitness_func=fitness, max_generations=100,cross_type='UNIFORM')
            #     avg_regular+=generations
            # avg_regular/=50
            # print("Bulls hit average generations to reach global optimum is: ", avg_bull_hit)
            # print("regular heuristc average generaions to reach global optimum is: ", avg_regular)
        if problem == 1:
            nqueens(100,100,0.1,0.3)
        if problem ==2:
            BinPackagingNDC.evolve()
        if problem == 3:
            BinPackagingNiche.evolve()
        if problem == 4:
            BinPackagingSpeciatin.evolve()
        if problem ==5:
            MathProblem.exaptation()
        if problem == 6:
            break


    # target = list("Hello, world!")
    # # Convert the string to bytes
    # my_bytes = ''.join(target).encode()
    #
    # # Convert the bytes to binary representation
    # my_binary = bin(int.from_bytes(my_bytes, byteorder='big'))
    # print(my_binary)


#############################
# import random
#
# # Encode the target string in binary form
# target = ''.join(format(ord(c), '08b') for c in "Hello, world!")
# num_genes = len(target)
#
#
# # Define the fitness function with "scrambler" heuristic
# def fitness_bull(individual):
#     score = 0
#     for i in range(num_genes):
#         if individual[i] == target[i]:
#             score += 1
#         elif individual[i] in target:
#              score += 0.5
#     return score
#
# def fitness(individual):
#     score = 0
#     for i in range(num_genes):
#         if individual[i] == target[i]:
#             score += 1
#         # elif individual[i] in target:
#         #     score += 0.5
#     return score
#
#
# # Define the genetic algorithm
# def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations):
#     # Initialize the population with random individuals
#     population = []
#     for i in range(pop_size):
#         individual = ''.join(str(random.randint(0, 1)) for j in range(num_genes))
#         population.append(individual)
#
#     # Evolve the population for a fixed number of generations
#     for generation in range(max_generations):
#         # Evaluate the fitness of each individual
#         fitnesses = [fitness_func(individual) for individual in population]
#
#         # Select the best individuals for reproduction
#         elite_size = int(pop_size * 0.1)
#         elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
#         elites = [population[i] for i in elite_indices]
#
#         # Generate new individuals by applying crossover and mutation operators
#         offspring = []
#         while len(offspring) < pop_size - elite_size:
#             parent1 = random.choice(elites)
#             parent2 = random.choice(elites)
#             child = ''.join(parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes))
#
#             # Apply mutation operator
#             if random.random() < 0.01:
#                 child = list(child)
#                 gene_index = random.randint(0, num_genes - 1)
#                 child[gene_index] = str(int(not int(child[gene_index])))
#                 child = ''.join(child)
#
#             offspring.append(child)
#         population = elites + offspring
#
#     # Find the individual with the highest fitness
#     best_individual = max(population, key=lambda individual: fitness_func(individual))
#     best_fitness = fitness_func(best_individual)
#
#     return best_individual, best_fitness
#
#
# best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=num_genes, fitness_func=fitness, max_generations=100)
# decoded_best_individual = ''.join(chr(int(best_individual[i:i+8], 2)) for i in range(0, num_genes, 8))
# print("Best individual:", decoded_best_individual)
# print("Best fitness:", best_fitness)
#
# # # Run the genetic algorithm with "scrambler" heuristic and print the result
# # avg_bull=0
# # avg_reg=0
# #
# # for i in range (100):
# #     best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=num_genes, fitness_func=fitness,
# #                                                   max_generations=100)
# #     avg_reg+=best_fitness
# #     decoded_best_individual = ''.join(chr(int(best_individual[i:i + 8], 2)) for i in range(0, num_genes, 8))
# # avg_reg/=100
# # for i in range (100):
# #     best_individual, best_fitness = genetic_algorithm(pop_size=100, num_genes=num_genes, fitness_func=fitness_bull,
# #                                                   max_generations=100)
# #     avg_bull+=best_fitness
# # avg_bull/=100
# # print("bull: ",avg_bull)
# # print("reg: ",avg_reg)
#
#
