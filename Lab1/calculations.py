import numpy as np


def genetic_diversification(population):
    '''
    * Hamming distanse between individuals
    * Number of different alleles in population 
    * Per generation
    '''
    dist=0
    for str1 in population:
        for str2 in population:
            dist+=hamming_distance(str1,str2)
    num_different_alleles = len(set(map(tuple, population)))
    print("--number of different alleles: {}, Distance between genes: {}".format(
        num_different_alleles, dist))
    pass

def selection_pressure(fitness_scores,k,size):
    '''
    Measuring Selection Pressure:
    The ratio between the probability that the most fit is selected to the probability 
    that the average member is selected.
    a. Fitness Variance
    b. Ratio as explained above 
    * Per generation
    '''
    # Calculate selection pressure
    selection_pressure = max(fitness_scores) - min(fitness_scores)
    # Calculate fitness variance
    fitness_variance = np.var(fitness_scores)
    # Calculate Top-Average Selection Probability Ratio (TASPR)
    selection_probabilities = [fitness_score / sum(fitness_scores) for fitness_score in fitness_scores]
    sorted_selection_probabilities = sorted(selection_probabilities, reverse=True)
    top_selection_probabilities_sum = sum(sorted_selection_probabilities[:k])
    average_selection_probability = 1 / size
    taspr = top_selection_probabilities_sum / (k * average_selection_probability)

    print("--Selection Pressure: {}, Fitness Variance: {}, Top-Average Selection Probability Ratio: {}".format(selection_pressure,fitness_variance, taspr))

    pass

def question_7():
    ''' 
    Test Nqueens and Bin packaging while:
    * Changing population size
    * Changing probability for mutations
    * Changing selection method
    * Changing survival strategy (Elitism, Aging)
    * Changing Crossover and mutation methods
    '''
    pass

def hamming_distance(string1, string2 ):
    # if len(string1) != len(string2):
    #     raise ValueException("Strings must be of equal length.")
    dist_counter = 0
    for i in range(len(string1)):
        if string1[i] != string2[i]:
            dist_counter += 1
    return dist_counter


def kendall_tau_distance(p, q):
    """
    Calculates the Kendall Tau distance between two permutations p and q
    """
    n = len(p)
    assert len(q) == n, "Permutations must be of equal length"

    inv_p = {v: k for k, v in enumerate(p)}
    inv_q = {v: k for k, v in enumerate(q)}

    # calculate the number of discordant pairs between the two permutations
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (p[i] < p[j] and q[inv_p[p[i]]] > q[inv_p[p[j]]]) or (p[i] > p[j] and q[inv_p[p[i]]] < q[inv_p[p[j]]]):
                count += 1

    return count
