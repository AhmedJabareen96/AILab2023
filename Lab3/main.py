import CVRP
import Point
import TabuSearch
from TabuSearch import *
from SimulatedAnnealing import *
from AntColony import *
from math import sqrt
from GA import *
from PSO import *

class args:
    maxIter = 400
    numNeighbors = 2048
    maxTabu = 20
    localOptStop = 25
    A = 3
    B = 4
    Q = 1000
    P = 0.1
    alpha = 0.5
    temperature = 100
names = ['0', 'E-n22-k4', 'E-n33-k4', 'E-n51-k5', 'E-n76-k10']
def get_input(input_file):
    file = open(names[int(input_file)] + '.txt', 'r')
    file.readline()# first three line contains no information
    file.readline()
    file.readline()

    dimension = file.readline()
    arr = [k for k in dimension.split(' ')]
    dimension = int(arr[2])

    file.readline()

    capacity = file.readline()
    arr = [k for k in capacity.split(' ')] # extract capacity
    capacity = int(arr[2])

    file.readline()

    wareHouse = file.readline()
    arr = [k for k in wareHouse.split(' ')] # extract cities coordinates
    wareHouse = Point.Point(int(arr[0]), int(arr[1]), int(arr[2])) # first city is the wareHouse

    cities = []
    for _ in range(dimension - 1):
        cityLine = file.readline()
        arr = [num for num in cityLine.split(' ')]
        city = Point.Point(int(arr[0]) - 1, int(arr[1]), int(arr[2])) # extract coordinantes
        cities.append(city)

    file.readline()
    file.readline()

    for i in range(dimension - 1):
        demandLine = file.readline()
        arr = [num for num in demandLine.split(' ')]
        cities[i].setDemand(int(arr[1]))

    cities.insert(0, wareHouse)

    distanceMat = calcDistanceMatrix(cities)
    cities.pop(0)

    ret = CVRP.CVRP(distanceMat, cities, capacity, len(cities))

    return ret


def calcDistanceMatrix(cities):
    arr1 = []
    for i in range(len(cities)):
        arr0 = []
        for j in range(len(cities)):
            arr0.append(distance(cities[i], cities[j]))
        arr1.append(arr0)
    return arr1


def distance(city1, city2):
    x = city1.x - city2.x
    dx = x ** 2

    y = city1.y - city2.y
    dy = y ** 2

    return sqrt(dx + dy)


def run():
    while True:
        print('Choose one of these problems :')
        input_file = input('- 0 : exercise example \n- 1 : E-n22-k4 \n- 2 : E-n33-k4 \n- 3 : E-n51-k5 \n- 4 : E-n76-k10 \n- 5 : EXIT\n')
        if int(input_file) == 5:
            break
        problem = get_input(input_file)
        print('Choose one of these  methods :')
        solution = int(input('- 1 : GA \n- 2 : ACO \n- 3 : PSO \n- 4 : Simulated Annealing \n- 5 : Tabu Search \n'))
        if solution == 1:
            genetic(problem)
        if solution == 2:
            ant_colony(problem)
        if solution == 3:
            pso(problem)
        if solution == 4:
            simulated_annealing(problem)
        if solution == 5 :
            tabu_search(problem)

        if int(input_file) == 5:
            break

def simulated_annealing(problem):
    simulatedAnnealing(problem, args())
    problem.printSolution()


def tabu_search(problem):
    tabuSearch_alg(problem, args())
    problem.printSolution()


def ant_colony(problem):
    antColonyopt(problem, args())
    problem.printSolution()


def genetic(problem):
    ga = GA(problem)
    ga.run()
    problem.printSolution()
def pso(problem):
    ga = PSO(problem)
    ga.run()
    problem.printSolution()



if __name__ == '__main__':
    run()

