import numpy as np

class SortingNetwork:
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.num_comparators = num_elements * (num_elements - 1) // 2
        self.comparators = np.zeros((self.num_comparators, 2), dtype=int)

    def crossover(self, other):
        child = SortingNetwork(self.num_elements)
        child.comparators = np.where(np.random.rand(*child.comparators.shape) < 0.5,
                                     self.comparators,
                                     other.comparators)
        return child

    def generate_random(self):
        self.comparators = np.random.randint(0, self.num_elements, size=(self.num_comparators, 2))

    def sort(self, input_vector):
        for i in range(self.num_comparators):
            a, b = self.comparators[i]
            if input_vector[a] > input_vector[b]:
                input_vector[a], input_vector[b] = input_vector[b], input_vector[a]
        return input_vector
