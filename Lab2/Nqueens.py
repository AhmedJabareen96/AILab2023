import ParentsSelection
from GAQueen import GAQueen
from random import random, shuffle, randint
import calculations
class NQueens:

    def __init__(self, N, POPSIZE, MAXITER, ELITRATE, MUTATION, pmx, mutateType,Parent_selection):
        self.N = N
        self.GA_MUTATION = MUTATION
        self.PMX = pmx
        self.Parent_selection=Parent_selection
        self.population = []
        self.nextPopulation = []
        self.mutateType = mutateType
        self.numElitation = POPSIZE * ELITRATE
        self.GA_ELITRATE = ELITRATE
        self.GA_POPSIZE = POPSIZE
        self.GA_MAXITER = MAXITER
        self.init_population()

    def init_population(self):
        for j in range(self.GA_POPSIZE):
            Str = [i for i in range(0, self.N)]
            Str1 = [i for i in range(0, self.N)]
            shuffle(Str)
            shuffle(Str1)
            member1 = GAQueen(self.N, Str)
            member2 = GAQueen(self.N, Str1)
            self.population.append(member1)
            self.nextPopulation.append(member2)


    def sort_by_fitness(self):
        self.population.sort(key=self.sort_helper)
    def sort_helper(self, x):
        return x.get_fitness()

    def calc_conflict(self, NQueens, j):
        conflicts = 0
        row = NQueens[j]
        col = j
        for i, k in zip(range(row), range(col)):
            if NQueens[k] == i:
                conflicts += 1

        for i, k in zip(range(row + 1, self.N), range(col)):
            if NQueens[col - 1 - k] == i:
                conflicts += 1

        for i, k in zip(range(row), range(col + 1, self.N)):
            if NQueens[k] == row - 1 - i:
                conflicts += 1

        for i, k in zip(range(row + 1, self.N), range(col + 1, self.N)):
            if NQueens[k] == i:
                conflicts += 1

        for i in range(self.N):
            if NQueens[i] == row and i != col:
                conflicts += 1
        return conflicts

    def calc_fitness(self):
        for i in range(self.GA_POPSIZE):
            fitness = 0
            for j in range(self.N):
                fitness += self.calc_conflict(self.population[i].NQueens, j)
            self.population[i].fitness = fitness / 2
    def print_reports(self,gen):
        print("Best Gene in generation ",gen," :",end=" ")
        for i in range(self.N):
            print(self.population[0].NQueens[i], end=" ")
        print(" , Fitness :", self.population[0].fitness)
        fitnessses = [obj.fitness for obj in self.population]
        strs = [obj.NQueens for obj in self.population ]
        calculations.selection_pressure(fitnessses,int(self.numElitation),self.GA_POPSIZE)
        calculations.genetic_diversification(strs)


    def pmx(self):
        for i in range(int(self.numElitation), self.GA_POPSIZE):
            i1=0
            i2=0
            fitnessses = [obj.fitness for obj in self.population]
            if self.Parent_selection == 0:
                i1 = randint(0, self.GA_POPSIZE / 2)
                i2 = randint(0, self.GA_POPSIZE / 2)
            if self.Parent_selection==1:
                i1 = ParentsSelection.roulette_wheel_selection(fitnessses)
                i2 = ParentsSelection.roulette_wheel_selection(fitnessses)
            if self.Parent_selection == 2:
                ind = ParentsSelection.stochastic_universal_sampling(fitnessses,self.GA_POPSIZE)
                i1 = ind[0]
                i2 = ind[1]
            i3 = randint(0, self.N-1)
            secondChar = self.population[i2].NQueens[i3]
            for j in range(1, self.N):
                self.nextPopulation[i].NQueens[j] = self.population[i1].NQueens[j]
            for j in range(self.N):
                if self.population[i1].NQueens[j] == secondChar:
                    self.nextPopulation[i].NQueens[j] = self.population[i1].NQueens[i3]
                    self.nextPopulation[i].NQueens[i3] = self.population[i1].NQueens[j]
                    break
            if self.mutateType == 0:
                if random() < self.GA_MUTATION:
                    self.inversion_mutation(i)
            else:
                if random() < self.GA_MUTATION:
                    self.Shuffle_mutation(i)

    def findFirstIndex(self, indicesArray):
        if len(indicesArray) == 0:
            return 0
        i = 1
        while i < self.N:
            if i not in indicesArray:
                return i
            i += 1
        return i

    def cx(self):
        for i in range(int(self.numElitation), self.GA_POPSIZE):
            i1 = randint(0, self.GA_POPSIZE-1)
            i2 = randint(0, self.GA_POPSIZE-1)
            parent1 = self.population[i1].NQueens
            parent2 = self.population[i2].NQueens
            indicesArray = []
            child = []
            odd = False
            while len(indicesArray) < self.N:
                firstIndex = self.findFirstIndex(indicesArray)
                index = firstIndex
                while True and firstIndex < self.N:
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
            self.nextPopulation[i].NQueens = child
            if self.mutateType == 0:
                if random() < self.GA_MUTATION:
                    self.inversion_mutation(i)
            else:
                if random() < self.GA_MUTATION:
                    self.Shuffle_mutation(i)

    def inversion_mutation(self,i):
            i1 = randint(0, self.N - 3)
            i2 = randint(i1 + 1, self.N - 2)
            i3 = randint(i2, self.N - 1)
            self.population[i].NQueens = self.population[i].NQueens[0:i1] + self.population[i].NQueens[i2:i3] + self.population[i].NQueens[i1:i2][::-1] + self.population[i].NQueens[i3:]

    def Shuffle_mutation(self,i):
            i1 = randint(0, self.N - 3)
            i2 = randint(i1 + 1, self.N - 2)
            str = self.population[i].NQueens[i1:i2]
            shuffle(str)
            self.population[i].NQueens = self.population[i].NQueens[0:i1] + str + self.population[i].NQueens[i2:]

    def mate(self):
        if self.PMX:
            self.pmx()
        else:
            self.cx()


    def swap(self):
        temp = self.population
        self.population = self.nextPopulation
        self.nextPopulation = temp

    def run(self):
        for index in range(self.GA_MAXITER):
            self.calc_fitness()
            self.sort_by_fitness()
            self.print_reports(index)
            if self.population[0].fitness == 0:
                break
            self.mate()
            self.swap()



