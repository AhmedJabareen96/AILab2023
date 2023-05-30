import random


class SortingNetwork:
    def __init__(self, N, minSize, maxSize):
        self.str = []
        self.fitness = 100000
        self.vectors = []
        self.init(N, minSize, maxSize)

    def evaluate(self):
        vectors = sum(self.checkSortingNetwork(vec) for vec in self.vectors)
        self.fitness = (len(self.vectors) - vectors) * 100 + len(self.str)

    def checkSortingNetwork(self, vec):
        vecTemp = list(vec.vector)

        for i in range(0, len(self.str), 2):
            i1 = vecTemp[self.str[i]]
            i2 = vecTemp[self.str[i + 1]]
            if int(i1) > int(i2):
                vecTemp[self.str[i]] = '0'
                vecTemp[self.str[i + 1]] = '1'
        for i in range(len(vecTemp) - 1):
            if vecTemp[i] > vecTemp[i + 1]:
                return 0
        return 1

    def init(self, N, minSize, maxSize):
        strSize = random.randint(minSize, maxSize)
        while strSize % 2 != 0:
            strSize = random.randint(minSize, maxSize)
        self.str = [random.randint(0, N - 1) for _ in range(strSize)]