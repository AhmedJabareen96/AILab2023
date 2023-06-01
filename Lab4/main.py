import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from coevolution import Coevolution

def main():
    print("Sorting Network")
    print("---------------")
    print("1. Network of 6 elements")
    print("2. Network of 16 elements")
    print("Press any other key to exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        num_elements = 6
    elif choice == "2":
        num_elements = 16
    else:
        return

    population_size = 100
    generations = 50
    input_vector = np.random.randint(0, 100, size=num_elements)

    ga = GeneticAlgorithm(num_elements, population_size, generations)
    best_fitness_scores = ga.evolve(input_vector)

    plt.plot(best_fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Genetic Algorithm - Sorting Network')
    plt.show()

    coevolution = Coevolution(num_elements, population_size, generations)
    best_network_fitness_scores, best_vector_fitness_scores = coevolution.evolve()

    plt.plot(best_network_fitness_scores, label='Best Network Fitness')
    plt.plot(best_vector_fitness_scores, label='Best Vector Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Coevolution - Sorting Network')
    plt.legend()
    plt.show()

    best_network = coevolution.network_population[np.argmax(best_network_fitness_scores)]
    best_vector = coevolution.vector_population[np.argmax(best_vector_fitness_scores)]
    best_sorted_vector = best_network.sort(best_vector)

    print("Best Vector:", best_vector)
    print("Best Sorting Network:")
    print(best_network)
    print("Optimal Number of Network Comparators:", best_network.num_comparators)

if __name__ == '__main__':
    main()
