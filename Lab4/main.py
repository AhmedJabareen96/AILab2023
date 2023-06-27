from coevolution import Coevolution
import numpy as np
import matplotlib.pyplot as plt


def main():
    num_elements = 0
    while True:
        choice = input("Enter the number of elements to be sorted:\n1. 6 elements\n2. 16 elements\nPress any other key to exit.\n")
        if choice == '1':
            num_elements = 6
            break
        elif choice == '2':
            num_elements = 16
            break
        else:
            return

    population_size = 500
    generations = 100

    coevolution = Coevolution(num_elements, population_size, generations)
    best_network_fitness_scores, best_vector_fitness_scores, comparator_counts = coevolution.evolve()

    # Print best fitness scores, best network, and best vector for each generation
    for generation in range(generations):
        print(f"Generation {generation+1}:")
        print(f"Best Sorting Network Fitness: {best_network_fitness_scores[generation]}")
        best_network = coevolution.network_population[np.argmax(best_network_fitness_scores[generation])]
        print(f"Best Network:\n{best_network.comparators}")
        print(f"Best Vector Fitness: {best_vector_fitness_scores[generation]}")
        best_vector = coevolution.vector_population[np.argmax(best_vector_fitness_scores[generation])]
        print(f"Best Vector:\n{best_vector}")
        print(f"Number of Comparators: {comparator_counts[generation]}")
        print()

    # Plotting fitness scores
    plt.plot(range(generations), best_network_fitness_scores, label='Network Fitness')
    plt.plot(range(generations), best_vector_fitness_scores, label='Vector Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Scores over Generations')
    plt.legend()
    plt.show()

    coevolution.network_population[10].draw()


if __name__ == '__main__':
    main()