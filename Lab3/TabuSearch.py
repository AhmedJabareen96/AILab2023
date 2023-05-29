import time
from random import randint

import Graph


def init_solution(size, problem):   # TSP: nearest neighbor heuristic
    city = randint(1, size)
    dictionary = {city: True}
    arr = []
    arr.append(city)
    index = size-1
    while index > 0:
        distances = problem.distanceMatrix[city]
        minCity = 1
        minDistance = float('inf')
        for i in range(1, len(distances)):
            distance = distances[i]
            if 0 < distance < minDistance and not dictionary.get(i, False):
                minDistance = distance
                minCity = i
        arr.append(minCity)
        dictionary[minCity] = True
        city = minCity
        index -= 1
    return arr


def getNeighborhood(bestCandidate, numNeighbors):
    return [mutate(bestCandidate) for _ in range(numNeighbors)]

def mutate(bestCandidate):
    return simpleInversionMutation(bestCandidate)

def exchangeMutation(sol):
    string = sol[:]
    i1 = randint(0, len(sol) - 1)
    i2 = randint(0, len(sol) - 1)
    string[i1], string[i2] = string[i2], string[i1]
    return string


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

def tabuSearch_alg(problem, args):
    startTime = time.time()
    local_counter = 0
    ret = []
    best = init_solution(problem.size, problem)
    bestFitness= problem.calcPathCost(best)
    bestCandidate = best
    globalBest = best
    globalFitness = bestFitness
    dict = {str(best): True}
    tabu = [best]
    gen=1
    for _ in range(args.maxIter):
        neighborhood = getNeighborhood(bestCandidate, args.numNeighbors)    # get neighborhood of current solution
        minimum= problem.calcPathCost(neighborhood[0])
        bestCandidate = neighborhood[0]
        for neighbor in neighborhood:   # search for the best neighbor
            cost = problem.calcPathCost(neighbor)
            if cost < minimum and not dict.get(str(neighbor), False):
                minimum = cost
                bestCandidate = neighbor
        if minimum < bestFitness:   # if found new best
            bestFitness = minimum
            best = bestCandidate
            local_counter = 0
        elif minimum == bestFitness:    # else incounter the local minimum number
            local_counter += 1
        if bestFitness < globalFitness: # update global minimum
            globalBest = best
            globalFitness = bestFitness
        tabu.append(bestCandidate)
        dict[str(bestCandidate)] = True
        if len(tabu) > args.maxTabu:
            dict[str(tabu[0])] = False
            tabu.pop(0)
        if local_counter == args.localOptStop:  # the case when stuck in local optimum ( reached the max number of local optimums)
            if bestFitness < globalFitness:
                globalBest = best
                globalFitness = bestFitness
            bestCandidate = init_solution(problem.size, problem)
            best = bestCandidate
            bestFitness= problem.calcPathCost(best)
            local_counter = 0
            dict = {str(bestCandidate): True}
        ret.append(min(bestFitness,globalFitness) )
        print('********* Generation :#',gen,' *************')
        print('best sol (permutation) = ', best)
        print('min cost = ', bestFitness)
        gen+=1
    print('Elapsed Time : ', time.time() - startTime)
    problem.best = globalBest
    Graph.draw(ret)
    problem.bestFitness = globalFitness