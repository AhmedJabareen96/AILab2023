class SNVector:
    def __init__(self, size, bnumber):
        self.vector = bnumber
        self.fitness = 10000
        self.networks = []
        self.size = size

    def evaluate(self):
        sortingNetworks = sum(self.checkSortingNetwork(network) for network in self.networks)
        self.fitness = len(self.networks) - sortingNetworks
        self.networks = []

    def checkSortingNetwork(self, network):
        vec = list(self.vector)  # Create a copy of the vector
        for i in range(0, len(network.str), 2):
            i1 = vec[network.str[i]]
            i2 = vec[network.str[i + 1]]
            if i1 > i2:
                vec[network.str[i]] = '0'
                vec[network.str[i + 1]] = '1'
        for i in range(len(vec) - 1):
            if vec[i] > vec[i + 1]:
                return 0
        return 1
