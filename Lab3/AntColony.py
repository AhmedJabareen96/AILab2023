from math import pow
from random import randint
from numpy.random import choice
import time
import Graph


def antColonyopt(problem, args):
    gr=[]
    bestPermutation, closest, currentBestPerm, currentBestclosest = reset()
    globalBest = []
    globalcloset = float('inf')
    local_counter = 0
    gen =1
    pheremonMatrix = [[float(1000) for _ in range(problem.size)] for _ in range(problem.size)]
    startTime = time.time()
    for _ in range(args.maxIter):
        tempPath = getPath(problem, pheremonMatrix, args)
        tempFitness= problem.calcPathCost(tempPath)
        if tempFitness < currentBestclosest:
            currentBestclosest = tempFitness
            currentBestPerm = tempPath
        if currentBestclosest < closest:  # check neighborhood
            closest = currentBestclosest
            bestPermutation = currentBestPerm
            local_counter = 0
        if currentBestclosest == closest:   # incounter local optimum counter if still stuck in the same place
            local_counter += 1
        if closest < globalcloset:# best solution
            globalBest = bestPermutation
            globalcloset = closest
        gr.append(globalcloset)
        updatePheremons(pheremonMatrix, tempPath, tempFitness, args.Q, args.P)
        print('********* Generation :#',gen,' *************')
        print('sol = ', bestPermutation)
        print('cost = ', closest)
        gen+=1
        if local_counter == args.localOptStop:  # reset every parameter if we stuck for a long time
            pheremonMatrix = [[float(1000) for _ in range(problem.size)] for _ in range(problem.size)]
            local_counter = 0
            bestPermutation,closest,currentBestPerm,currentBestclosest=reset()
    print('Time elapsed: ', time.time() - startTime)
    Graph.draw(gr)
    problem.best = globalBest   # save the solution and its fitness
    problem.bestFitness = globalcloset


def check_new_route(tempFitness,currentBestPath,tempPath,bestFitness,currentBestFitness,bestPath,globalFitness):
    if tempFitness < currentBestFitness:
        currentBestFitness = tempFitness
        currentBestPath = tempPath
    if currentBestFitness < bestFitness:  # update best (take a step towards the better neighbor)
        bestFitness = currentBestFitness
        bestPath = currentBestPath
        local_counter = 0
    if currentBestFitness == bestFitness:  # to detect local optimum
        local_counter += 1
    if bestFitness < globalFitness:  # update the best solution found untill now
        globalBest = bestPath
        globalFitness = bestFitness
    return tempFitness,currentBestPath,tempPath,bestFitness,currentBestFitness,bestPath,globalFitness
def reset():
    bestPath = []
    bestFitness = float('inf')
    currentBestPath = []
    currentBestFitness = float('inf')
    return  bestPath,bestFitness,currentBestPath,currentBestFitness
def getPath(problem, pheremonMatrix, args):
    cities = [i + 1 for i in range(problem.size)]
    probMatrix = []
    for i in range(problem.size):
        arrayProb = []
        for j in range(problem.size):
            num = 0
            if i != j:
                num = calculateProb(i, j, pheremonMatrix, problem.distanceMatrix, args.A, args.B)
            arrayProb.append(num)
        probMatrix.append(arrayProb)
    currentCity = randint(1, len(cities))
    cities[currentCity - 1] = -1
    path = [currentCity]
    while len(cities) != len(path):
        probVect = getProbVector(currentCity, probMatrix)
        updateProbMatrix(currentCity, probMatrix)
        currentCity = choice(cities, p=probVect)
        cities[currentCity - 1] = -1
        path.append(currentCity)
    return path


def calculateProb(i, j, pheremonMatrix, distanceMatrix, a, b):
    tau = float(pheremonMatrix[i][j])
    n = float(1 / float(distanceMatrix[i + 1][j + 1]))
    return float(pow(tau, a) * pow(n, b))


def getProbVector(currentCity, probMatrix):
    vector = []
    arr = probMatrix[currentCity - 1][:]
    probSum = sum(arr)
    for i in range(len(probMatrix[0])):
        vector.append(float(float(probMatrix[currentCity - 1][i]) / float(probSum)))
    return vector


def updateProbMatrix(currentCity, probMatrix):
    for i in range(len(probMatrix[0])):
        probMatrix[currentCity - 1][i] = float(0)
        probMatrix[i][currentCity - 1] = float(0)


def updatePheremons(pheremonMatrix, bestPath, pathCost, q, p):
    currentPheremons = [[float(0) for _ in range(len(pheremonMatrix[0]))] for _ in range(len(pheremonMatrix[0]))]
    for i in range(len(bestPath) - 1):
        currentPheremons[bestPath[i] - 1][bestPath[i + 1] - 1] = float(float(q) / float(pathCost))
        currentPheremons[bestPath[i + 1] - 1][bestPath[i] - 1] = float(float(q) / float(pathCost))

    for i in range(len(pheremonMatrix[0])):
        for j in range(len(pheremonMatrix[0])):
            num1 = float(pheremonMatrix[i][j] * (1 - p))
            num2 = float(currentPheremons[i][j] * p)
            pheremonMatrix[i][j] = float(num1 + num2)
            if pheremonMatrix[i][j] < 0.0001:
                pheremonMatrix[i][j] = 0.0001