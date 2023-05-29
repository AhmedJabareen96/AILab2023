class Island:
    def __init__(self, CVRP, population_size):
        self.population = []
        self.buffer = []
        self.CVRP = CVRP
        self.init_population(population_size)
        self.migration_rate = 0.1
        self.immigrant_rate = 0.1

    def init_islands(self):
         for i in range(5):  # Create 5 islands
            island = Island(self.CVRP, popsize)
            self.islands.append(island)

    def migrate_between_islands(self):
        num_islands = len(self.islands)
        for i, island in enumerate(self.islands):
            if i < num_islands - 1:
                next_island = self.islands[i + 1]
                migrants = int(island.migration_rate * popsize)
                for j in range(migrants):
                    index = randint(0, popsize - 1)
                    migrant = island.population[index]
                    next_island.receive_migrant(migrant)
    def init_population(self, population_size):
        for _ in range(population_size):
            rand_str1 = init_sol(self.CVRP)
            rand_str2 = init_sol(self.CVRP)
            member1 = GAstruct(rand_str1, 0)
            member2 = GAstruct(rand_str2, 0)
            self.population.append(member1)
            self.buffer.append(member2)

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def fitness_sort(self, x):
        return x.fitness

    def calc_fitness(self):
        for i in range(popsize):
            fitness, _ = self.CVRP.calcPathCost(self.population[i].str)
            self.population[i].fitness = fitness

    def pmx(self):
        esize = popsize * elite_rate
        size = self.CVRP.size
        for i in range(int(esize), popsize):
            i1 = randint(0, popsize // 2)
            i2 = randint(0, popsize // 2)
            i3 = randint(0, size - 1)
            second_citizen_queen = self.population[i2].str[i3]
            for j in range(1, size):
                self.buffer[i].str[j] = self.population[i1].str[j]
            for j in range(size):
                if self.population[i1].str[j] == second_citizen_queen:
                    self.buffer[i].str[j] = self.population[i1].str[i3]
                    self.buffer[i].str[i3] = self.population[i1].str[j]
                    break
            if random() < mutation:
                self.random_mutation(i)
        self.swap()

    def random_mutation(self, i):
        size = self.CVRP.size
        member = self.buffer[i].str
        i1 = randint(0, size - 3)
        i2 = randint(i1 + 1, size - 2)
        new_hdak = member[i1:i2]
        shuffle(new_hdak)
        self.buffer[i].str = member[0:i1] + new_hdak + member[i2:]

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def receive_migrant(self, migrant):
        self.buffer.append(migrant)

    def immigrate(self):
        num_immigrants = int(self.immigrant_rate * popsize)
        immigrants = []
        for i in range(num_immigrants):
            rand_str = init_sol(self.CVRP)
            immigrant = GAstruct(rand_str, 0)
            immigrants.append(immigrant)
        self.population.extend(immigrants)
        self.population = self.population[:popsize]




import time
from random import random, shuffle, randint, choice
import Graph
import numpy

popsize = 2048
size = 10
elite_rate = 0.1
mutation = random() * 0.25
maxIter = 200


def init_sol(problem):  # TSP: nearest neighbor heuristic
    size = problem.size
    array = []
    city = randint(1, size)
    array.append(city)
    dictionary = {city: True}
    index = size-1
    while index > 0:
        distanceArray = problem.distanceMatrix[city]
        minCity = 1
        minDistance = float('inf')
        for i in range(1, len(distanceArray)):
            distance = distanceArray[i]
            if 0 < distance < minDistance and not dictionary.get(i, False):
                minDistance = distance
                minCity = i
        array.append(minCity)
        dictionary[minCity] = True
        city = minCity
        index -= 1
    return array


class GAstruct:
    def __init__(self, string, fitness):
        self.str = string
        self.fitness = fitness
        self.specynum=0


class GA:
    def __init__(self, CVRP):
        self.population = []
        self.buffer = []
        self.CVRP = CVRP
        self.init_population()
        self.threshhold=0.5
        self.species=[]

    def init_population(self):
        for i in range(popsize):
            randStr1 = init_sol(self.CVRP)
            randStr2 = init_sol(self.CVRP)
            member1 = GAstruct(randStr1, 0)
            member2 = GAstruct(randStr2, 0)
            self.population.append(member1)
            self.buffer.append(member2)

    def sort_by_fitness(self):
        self.population.sort(key=self.fitness_sort)

    def fitness_sort(self, x):
        return x.fitness

    def distance(self, firstIndex, secondIndex):
        # values1 = self.population[firstIndex]
        # values2 = self.population[secondIndex]
        """Compute the Kendall tau distance."""
        values1 = firstIndex.NQueens
        values2 = secondIndex.NQueens
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = numpy.meshgrid(numpy.arange(n), numpy.arange(n))
        a = numpy.argsort(values1)
        b = numpy.argsort(values2)
        ndisordered = numpy.logical_or(numpy.logical_and(a[i] < a[j], b[i] > b[j]),
                                       numpy.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        # print(ndisordered / (n * (n - 1)))
        return ndisordered / (n * (n - 1))


    def Threshhold_spec(self):
        numberof_species = -1
        for i in range(self.GA_POPSIZE):
            flag = 0
            for j in range(len(self.species)):
                counter = 0
                for k in range(len(self.species[j])):
                    destence = self.distance(self.population[i], self.species[j][k])
                    if destence < self.threshhold:
                        counter += 1
                if counter == len(self.species[j]):
                    flag = 1
                    self.population[i].specynum = j
                    self.species[j].append(self.population[i])
                    break
            if flag == 0:
                new_array = []
                self.population[i].specynum = len(self.species)
                new_array.append(self.population[i])
                self.species.append(new_array)
                numberof_species += 1





    def calc_fitness(self):
        for i in range(popsize):
            fitness = self.CVRP.calcPathCost(self.population[i].str)
            fitness2 = self.CVRP.calcPathCost(self.buffer[i].str)
            arr = self.population[i].str
            arr2 = self.buffer[i].str

            for j in range(len(arr)):
                if j + 1 not in arr:
                    fitness += 1000
            for j in range(len(arr)):
                if j + 1 not in arr2:
                    fitness2 += 1000
            self.population[i].fitness = fitness
            self.buffer[i].fitness = fitness2

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def pmx(self):
        esize = popsize * elite_rate
        size = self.CVRP.size
        for i in range(int(esize), popsize):
            i1 = randint(0, popsize / 2)
            i2 = randint(0, popsize / 2)
            i3 = randint(0, size - 1)
            secondCitizenQueen = self.population[i2].str[i3]
            for j in range(1, size):
                self.buffer[i].str[j] = self.population[i1].str[j]
            for j in range(size):
                if self.population[i1].str[j] == secondCitizenQueen:
                    self.buffer[i].str[j] = self.population[i1].str[i3]
                    self.buffer[i].str[i3] = self.population[i1].str[j]
                    break
            if random() < mutation:
                self.random_mutation(i)
        self.swap()

    def findFirstIndex(self, indicesArray):
        size = self.CVRP.size
        if len(indicesArray) == 0:
            return 0
        i = 1
        while i < size:
            if i not in indicesArray:
                return i
            i += 1
        return i

    def cx(self):
        esize = popsize * elite_rate
        size = self.CVRP.size
        for i in range(int(esize), popsize):
            i1 = randint(0, popsize - 1)
            i2 = randint(0, popsize - 1)
            parent1 = self.population[i1].str
            parent2 = self.population[i2].str
            indicesArray = []
            child = []
            odd = False
            while len(indicesArray) < size:
                firstIndex = self.findFirstIndex(indicesArray)
                index = firstIndex
                while True and firstIndex < size:
                    indicesArray.append(index)
                    if odd:
                        child.append(parent1[index])
                        index = parent1.index(parent2[index])
                    else:
                        child.append(parent2[index])
                        index = parent1.index(parent2[index])
                    if index == firstIndex:
                        break
                odd = not odd
            self.buffer[i].str = child
            if random() < mutation:
                self.random_mutation(i)
        self.swap()

    def random_immigrant(self):
        esize = popsize * 0.9
        for i in range(int(esize), popsize):
            i1 = randint(0, int(esize))
            self.population[i] = self.population[i1]
            if random() < mutation:
                self.random_mutation(i)

    def random_mutation(self, i):
        # take part of the array shuffle it put it back
        size = self.CVRP.size
        member = self.buffer[i].str
        i1 = randint(0, size - 3)
        i2 = randint(i1 + 1, size - 2)
        newHdak = member[i1:i2]
        shuffle(newHdak)
        self.buffer[i].str = member[0:i1] + newHdak + member[i2:]

    def mate(self):
        esize = popsize * elite_rate
        size = self.CVRP.size
        for i in range(int(esize), popsize):
            i1 = randint(0, popsize / 2)
            i2 = randint(0, popsize / 2)
            spos = randint(0, size - 1)
            self.buffer[i].str = self.population[i1].str[0:spos] + self.population[i2].str[spos:]
            if random() < mutation:
                self.mutate(i)
        self.swap()

    def mutate(self, i):
        i1 = randint(0, self.CVRP.size - 1)
        i2 = randint(0, self.CVRP.size - 1)
        tmp = self.buffer[i].str[i1]
        self.buffer[i].str[i1] = self.buffer[i].str[i2]
        self.buffer[i].str[i2] = tmp

    def print_best(self,iter):
        print('********* generation :#',iter,' ************')
        print("sol = ", self.population[0].str)
        print('cost = ', self.population[0].fitness)
        print()

    def run(self):
        gr = []
        startTime = time.time()
        for i in range(maxIter):
            self.calc_fitness()
            self.sort_by_fitness()
            self.pmx()
            self.CVRP.best = self.population[0].str
            self.CVRP.bestFitness = self.population[0].fitness
            self.print_best(i)
            if len(gr)>0 and self.population[0].fitness<= min(gr):
                gr.append(self.population[0].fitness)
            if i==0:
                gr.append(self.population[0].fitness)
        Graph.draw(gr)
        print('Time elapsed: ', time.time() - startTime)