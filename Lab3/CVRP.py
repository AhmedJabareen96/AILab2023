class CVRP:
    def __init__(self, distanceMatrix, cities, capacity, size):
        self.cities = cities
        self.capacity = capacity
        self.size = size
        self.best = []
        self.bestFitness = 0
        self.distanceMatrix = distanceMatrix
    def calcPathCost(self, path):
        left = self.capacity
        overall = 0
        overall += self.distanceMatrix[path[0]][0]
        left -= self.cities[path[0] - 1].capacity
        for i in range(0,len(path)-1):
            city1 = path[i]     # looking to the current 2 cities in the permutuaion
            city2 = path[i + 1]
            if self.cities[city2 - 1].capacity <= left: # the veichle can still apply to the next demand so no need to a new one
                left -= self.cities[city2 - 1].capacity
                overall += self.distanceMatrix[city1][city2]
            else: # new veichle is needed
                overall += self.distanceMatrix[city1][0] #move the last veichle to the ware house from the current point
                left = self.capacity # refill capacity because we have a new empty veichle
                overall += self.distanceMatrix[city2][0] # add the first distance for the new veichle
                left -= self.cities[city2 - 1].capacity # subtract the first demand from capcity
        overall += self.distanceMatrix[path[len(path)-1]][0] # return last veichle to the ware house
        return overall

    def pathToVehicles(self, path):
        capacity = self.capacity
        index = 0
        capacity -= self.cities[path[index] - 1].capacity
        pivot = 0
        subArrays = []
        while index < (len(path) - 1):
            city = path[index + 1]
            if self.cities[city - 1].capacity <= capacity:
                capacity -= self.cities[city - 1].capacity
            else:
                subArrays.append(path[pivot : index + 1])
                pivot = index + 1
                capacity = self.capacity
                capacity -= self.cities[city - 1].capacity
            index += 1

        subArrays.append(path[pivot : len(path)])

        for arr in subArrays:
            arr.insert(0, 0)
            arr.append(0)

        return subArrays

    def printSolution(self):
        paths = self.pathToVehicles(self.best)
        i=1
        for path in paths:
            print("route #",i,":  ", *path)
            i+=1
        print("Cost : ",        self.bestFitness )