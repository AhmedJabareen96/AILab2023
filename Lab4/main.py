import matplotlib.pyplot as plt
from coevolution import Coevolution

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

    population_size = 50
    generations = 100

    coevolution = Coevolution(num_elements, population_size, generations)
    best_network_fitness_scores, best_vector_fitness_scores = coevolution.evolve()

    # Plotting fitness scores
    plt.plot(range(generations), best_network_fitness_scores, label='Network Fitness')
    plt.plot(range(generations), best_vector_fitness_scores, label='Vector Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Scores over Generations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
