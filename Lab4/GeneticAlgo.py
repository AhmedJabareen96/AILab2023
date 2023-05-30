from random import randint, random, choices, sample
import Graph
from SortingNetwork import SortingNetwork
from SNVector import SNVector

GROUP_SIZE = 5000


class GenStruct:
    def __init__(self, N, arrSize):
        self.arr = [randint(1, N) for _ in range(arrSize)]
        self.fitness = -1


class GeneticAlgo:
    def __init__(self, N, minCN, maxCN, maxIter):
        self.popSize = 2048
        self.population = []
        self.vectors = []
        self.buffer = []
        self.N = N
        self.maxIter = maxIter
        self.minCN = minCN
        self.mutation = random() * 0.25
        self.arrSize = minCN
        self.maxCN = maxCN
        self.initPopulation()
        self.initVec()

    def initPopulation(self):
        self.population = [SortingNetwork(self.N, self.minCN, self.maxCN) for _ in range(self.popSize)]
        self.buffer = [SortingNetwork(self.N, self.minCN, self.maxCN) for _ in range(self.popSize)]

    def initVec(self):
        self.vectors = [SNVector(self.N, list(format(i, '0{:d}b'.format(self.N)))) for i in range(2 ** self.N)]

    def calcFitness(self):
        for member in self.population:
            member.evaluate()
        for vector in self.vectors:
            vector.evaluate()

    def mate(self):
        esize = int(self.popSize * 0.1)
        for i in range(esize, self.popSize):
            i1 = randint(0, int(self.popSize / 2))
            i2 = randint(0, int(self.popSize / 2))
            spos = randint(0, self.arrSize - 1)
            self.buffer[i].str = self.population[i1].str[:spos] + self.population[i2].str[spos:]

            if random() < self.mutation:
                self.mutate(i)
        self.swap()

    def mutate(self, i):
        i1 = randint(0, self.arrSize - 2)
        while i1 % 2 == 1:
            i1 = randint(0, self.arrSize - 2)
        i2 = randint(0, self.arrSize - 2)
        while i2 % 2 == 1:
            i2 = randint(0, self.arrSize - 2)

        self.buffer[i].str[i1], self.buffer[i].str[i2] = self.buffer[i].str[i2], self.buffer[i].str[i1]
        self.buffer[i].str[i1 + 1], self.buffer[i].str[i2 + 1] = self.buffer[i].str[i2 + 1], self.buffer[i].str[i1 + 1]

    def swap(self):
        self.population, self.buffer = self.buffer, self.population

    def printBest(self):
        best_member = self.population[0]
        print("SOL =", best_member.str)
        print('FITNESS =', best_member.fitness)
        print('ARRAY SIZE =', len(best_member.str))
        print()

    def sortByFitness(self):
        self.population.sort(key=lambda x: x.fitness)
        self.vectors.sort(key=lambda x: x.fitness)

    def assign(self, iteration):
        if self.N < 10:
            for member in self.population:
                member.vectors = self.vectors
            for i in range(len(self.vectors)):
                self.vectors[i].networks = self.population
        else:
            if iteration == 0:
                for member in self.population:
                    member.vectors = [self.vectors[j] for j in sample(range(len(self.vectors)), k=GROUP_SIZE)]
                    for vector in member.vectors:
                        vector.networks.append(member)
            else:
                for member in self.population:
                    member.vectors = [self.vectors[j] for j in sample(range(int(len(self.vectors) / 2)), k=GROUP_SIZE)]
                    for vector in member.vectors:
                        vector.networks.append(member)

    def run(self):
        fitnessArray = []
        sizeArray = []
        for i in range(self.maxIter):
            self.assign(i)
            self.calcFitness()
            self.sortByFitness()
            print("ITERATION:", i, '/', self.maxIter)
            self.printBest()
            fitnessArray.append(self.population[0].fitness)
            sizeArray.append(len(self.population[0].str))
            if self.population[0].fitness == self.minCN:
                break
            self.mate()
        Graph.draw(fitnessArray)
        Graph.draw(sizeArray, "Array Size")



# N = 10
# minCN = 5
# maxCN = 20
# maxIter = 100

# genetic_algo = GeneticAlgo(N, minCN, maxCN, maxIter)
# genetic_algo.run()
