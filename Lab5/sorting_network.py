import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class SortingNetwork:
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.num_comparators = num_elements * (num_elements - 1) // 2
        self.generate_random()

    def crossover(self, other):
        child = SortingNetwork(self.num_elements)
        child.comparators = np.where(np.random.rand(*child.comparators.shape) < 0.5,
                                     self.comparators,
                                     other.comparators)
        return child

    def copy(self):
        new_network = SortingNetwork(self.num_elements)
        new_network.comparators = np.copy(self.comparators)
        return new_network
    def generate_random(self):
        self.comparators = np.random.randint(0, self.num_elements, size=(self.num_comparators, 2))
        for comp in self.comparators:
            np.sort(comp)

    def sort(self, input_vector):
        for k in range(len(input_vector)):
            for i in range(len(self.comparators)):
                a, b = self.comparators[i]
                if input_vector[k][a] > input_vector[k][b]:
                    input_vector[k][a], input_vector[k][b] = input_vector[k][b], input_vector[k][a]
        return input_vector
    def delete_comp(self,index):
        # Delete the element at the random index
        self.comparators = np.delete(self.comparators, index, axis=0)
        self.num_comparators-=1
    def add_comp(self):
        new_comparator = np.random.randint(0, self.num_elements, size=(1, 2))
        new_comparator=np.sort(new_comparator)
        self.comparators = np.append(self.comparators, new_comparator, axis=0)
        self.num_comparators += 1
    def draw(self):
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes for each index
        for i in range(self.num_elements):
            graph.add_node(i)

        # Add edges between rows based on comparators
        for i in range(len(self.comparators)):
            a, b = self.comparators[i]
            graph.add_edge(a, b)

        # Set the position of each node
        pos = {}
        for i in range(self.num_elements):
            pos[i] = (0, i)

        # # Draw nodes
        # nx.draw_networkx_nodes(graph, pos, node_color='blue')

        # Draw horizontal lines for each index
        for i in range(self.num_elements):
            plt.plot([0, 1000 ], [i, i], 'k-', linewidth=2)

        # Draw vertical lines for comparators
        x_offset = 1
        y_offset = 0.5
        for i in range(len(self.comparators)):
            a, b = self.comparators[i]
            nx.draw_networkx_edges(graph, pos, edgelist=[(a, b)], edge_color='black', width=2.0,
                                   arrows=False, connectionstyle=f"arc3, rad={0.2 + y_offset / 6}")
            plt.plot([x_offset, x_offset], [a, b], 'k-', linewidth=2)
            x_offset += 1

        # Set axis properties
        plt.title("Sorting Network")
        plt.xlim(0, x_offset)
        plt.ylim(-1, self.num_elements)
        plt.axis('off')

        # Show the plot
        plt.show()