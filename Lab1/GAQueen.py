class GAQueen:
    def __init__(self, string, fitness):
        self.N = string
        self.NQueens = fitness
        self.fitness = 0
        self.age = 0

    def get_str(self):
        return self.str

    def get_fitness(self):
        return self.fitness

    def set_str(self, string):
        self.str = string

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __gt__(self, other):
        return self.getFitness() >= other.getFitness()

    def __lt__(self, other):
        return self.getFitness() < other.getFitness()
