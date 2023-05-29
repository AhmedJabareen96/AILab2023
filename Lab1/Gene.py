class Gene:
    def __init__(self, str, fitness, age):
        self.str = str
        self.fitness = fitness
        self.age = age
        self.rank = None

    def getString(self):
        return self.str

    def getFitness(self):
        return self.fitness

    def setString(self, string):
        self.str = string

    def setFitness(self, fitness):
        self.fitness = fitness

