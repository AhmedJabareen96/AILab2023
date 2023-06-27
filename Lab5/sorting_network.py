import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


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
    def draw(self):
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes for each index
        for i in range(self.num_elements):
            graph.add_node(i)

        # Add edges between rows based on comparators
        for i in range(self.num_comparators):
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
        for i in range(self.num_comparators):
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