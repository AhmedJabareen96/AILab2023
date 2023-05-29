import math

def manhattan_distance(str1, str2):
    assert len(str1) == len(str2), "Strings must have the same length"
    distance = 0
    for i in range(len(str1)):
        distance += abs(ord(str1[i]) - ord(str2[i]))
    return distance


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("The strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def euclidean_distance(str1, str2):
    assert len(str1) == len(str2), "Strings must have the same length"
    squared_diffs = 0
    for i in range(len(str1)):
        squared_diffs += (ord(str1[i]) - ord(str2[i]))**2
    return math.sqrt(squared_diffs)


def kendall_tau_distance(str1, str2):
    assert len(str1) == len(str2), "Strings must have the same length"
    diffs = 0
    for i in range(len(str1)):
        for j in range(i+1, len(str1)):
            if (str1[i] < str1[j] and str2[i] > str2[j]) or (str1[i] > str1[j] and str2[i] < str2[j]):
                diffs += 1
    return diffs
