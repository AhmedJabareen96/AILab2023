from random import randint
from numpy import exp
from numpy.random import rand
import time

import Graph


def init_sol(size, problem):
    city = randint(1, size)
    arr= []
    arr.append(city)
    dict= {city: True}
    index = size-1
    while index > 0:
        distanceArray = problem.distanceMatrix[city]
        minCity = 1
        minDistance = float('inf')
        for i in range(1, len(distanceArray)):
            distance = distanceArray[i]
            if 0 < distance < minDistance and not dict.get(i, False):
                minDistance = distance
                minCity = i
        arr.append(minCity)
        dict[minCity] = True
        city = minCity
        index -= 1
    return arr


def getNeighborhood(bestCandidate, numNeighbors):
    return [mutate(bestCandidate) for _ in range(numNeighbors)]


def mutate(bestCandidate):
    return simpleInversionMutation(bestCandidate)



def simpleInversionMutation(sol):
    string = sol[:]
    i1 = randint(0, len(sol) - 1)
    i2 = randint(0, len(sol) - 1)
    if i1 > i2:
        i1, i2 = i2, i1
    while i1 < i2:
        string[i1], string[i2] = string[i2], string[i1]
        i1 += 1
        i2 -= 1
    return string


def simulatedAnnealing(problem, args):
    neighbors = 14
    startTime = time.time()
    gr = []
    best = init_sol(problem.size, problem)
    bestFitness = problem.calcPathCost(best)
    globalBest = best
    globalFitness = bestFitness
    currentBest = best
    currentFitness = bestFitness
    temperature = float(args.temperature)
    counter = 0
    gen = 1
    for _ in range(args.maxIter):
        neighborhood = getNeighborhood(best, args.numNeighbors)
        for _ in range(neighbors):  # search in the some neighbors and find the best
            randNeighbor = neighborhood[randint(0, len(neighborhood) - 1)]
            neighborFitness = problem.calcPathCost(randNeighbor)
            diff = neighborFitness - bestFitness
            metropolis = float(exp(float(-1 * diff) / temperature))
            if neighborFitness < currentFitness or rand() < metropolis:
                currentFitness = neighborFitness
                currentBest = randNeighbor
        if currentFitness < bestFitness:
            best = currentBest
            bestFitness = currentFitness
            counter = 0
        if currentFitness == bestFitness:
            counter += 1
        if bestFitness < globalFitness:
            globalBest = best
            globalFitness = bestFitness
        if counter == args.localOptStop:
            best = init_sol(problem.size, problem)
            bestFitness= problem.calcPathCost(best)
            currentBest = best
            currentFitness = bestFitness
            counter = 0
            temperature = float(args.temperature)
        print('********* Generation :#',gen,' *************')
        print('best sol (permutation) = ', best)
        print('min cost = ', bestFitness)
        gen+=1
        gr.append(globalFitness)
        temperature *= args.alpha
    print('Time elapsed: ', time.time() - startTime)
    problem.best = globalBest
    problem.bestFitness = globalFitness
    Graph.draw(gr)


def reset():
    bestPath = []
    bestFitness = float('inf')
    currentBestPath = []
    currentBestFitness = float('inf')
    return  bestPath,bestFitness,currentBestPath,currentBestFitness

import random
import numpy as np


class Solution:
    def __init__(self, num_customers, num_vehicles, depot, distance_matrix):
        self.routes = [[] for _ in range(num_vehicles)]
        self.current_capacity = [0] * num_vehicles
        self.depot = depot
        self.distance_matrix = distance_matrix
        self.total_distance = self.calculate_total_distance()

    def calculate_total_distance(self):
        total_distance = 0
        for route in self.routes:
            if route:
                current_node = self.depot
                for customer in route:
                    total_distance += self.distance_matrix[current_node - 1][customer - 1]
                    current_node = customer
                total_distance += self.distance_matrix[current_node - 1][self.depot - 1]
        return total_distance

    def insert_customer(self, customer, vehicle):
        self.routes[vehicle].append(customer)
        self.current_capacity[vehicle] += 1
        self.total_distance = self.calculate_total_distance()

    def remove_customer(self, customer, vehicle):
        self.routes[vehicle].remove(customer)
        self.current_capacity[vehicle] -= 1
        self.total_distance = self.calculate_total_distance()


def generate_initial_solution(num_customers, num_vehicles, depot, distance_matrix):
    solution = Solution(num_customers, num_vehicles, depot, distance_matrix)
    unassigned_customers = list(range(1, num_customers + 1))
    for vehicle in range(num_vehicles):
        while unassigned_customers and solution.current_capacity[vehicle] < num_customers // num_vehicles:
            customer = random.choice(unassigned_customers)
            solution.insert_customer(customer, vehicle)
            unassigned_customers.remove(customer)
    return solution


def get_neighborhood(solution):
    neighborhood = []
    for vehicle in range(len(solution.routes)):
        for i, customer in enumerate(solution.routes[vehicle]):
            new_solution = Solution(len(solution.routes), len(solution.routes[0]), solution.depot, solution.distance_matrix)
            new_solution.routes = [route.copy() for route in solution.routes]
            new_solution.current_capacity = solution.current_capacity.copy()
            new_solution.remove_customer(customer, vehicle)
            for new_vehicle in range(len(solution.routes)):
                if new_solution.current_capacity[new_vehicle] < len(solution.routes[0]):
                    new_solution.insert_customer(customer, new_vehicle)
                    neighborhood.append(new_solution)
    return neighborhood


def acceptance_probability(current_fitness, new_fitness, temperature):
    if new_fitness < current_fitness:
        return 1.0
    return np.exp((current_fitness - new_fitness) / temperature)


def simulated_annealing(num_iterations, num_customers, num_vehicles, depot, distance_matrix, initial_temperature, cooling_rate):
    current_solution = generate_initial_solution(num_customers, num_vehicles, depot, distance_matrix)
    current_fitness = current_solution.total_distance
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temperature

    for _ in range(num_iterations):
        neighborhood = get_neighborhood(current_solution)
        new_solution = random.choice(neighborhood)
        new_fitness = new_solution.total_distance

        if acceptance_probability(current_fitness, new_fitness, temperature) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness

        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        temperature *= cooling_rate

    return best_solution.routes, best_fitness