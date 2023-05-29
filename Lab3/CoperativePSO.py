import random
import numpy as np


class Particle:
    def __init__(self, num_customers, num_vehicles, depot, distance_matrix):
        self.position = np.random.permutation(range(1, num_customers + 1))
        self.velocity = np.zeros_like(self.position)
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = float('inf')
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.distance_matrix = distance_matrix

    def update_velocity(self, gbest_position, omega, phi_p, phi_g):
        r_p = np.random.rand(*self.position.shape)
        r_g = np.random.rand(*self.position.shape)
        self.velocity = (omega * self.velocity +
                         phi_p * r_p * (self.pbest_position - self.position) +
                         phi_g * r_g * (gbest_position - self.position))

    def update_position(self):
        self.position = np.clip(self.position + self.velocity, 1, self.num_vehicles)

    def evaluate_fitness(self):
        routes = self.construct_routes()
        total_distance = self.calculate_total_distance(routes)
        self.pbest_fitness = total_distance

    def construct_routes(self):
        routes = [[] for _ in range(self.num_vehicles)]
        current_capacity = [0] * self.num_vehicles
        for customer in self.position:
            assigned = False
            for idx, route in enumerate(routes):
                if self.check_capacity(route, current_capacity[idx], customer):
                    route.append(customer)
                    current_capacity[idx] += 1
                    assigned = True
                    break
            if not assigned:
                routes[random.randint(0, self.num_vehicles - 1)].append(customer)
        return routes

    def calculate_total_distance(self, routes):
        total_distance = 0
        for route in routes:
            if route:
                current_node = self.depot
                for customer in route:
                    total_distance += self.distance_matrix[current_node - 1][customer - 1]
                    current_node = customer
                total_distance += self.distance_matrix[current_node - 1][self.depot - 1]
        return total_distance

    def check_capacity(self, route, current_capacity, customer):
        if current_capacity == 0:
            return True
        return current_capacity + 1 <= self.num_vehicles and current_capacity + 1 <= len(route)


def cooperative_pso(num_particles, num_iterations, num_customers, num_vehicles, depot, distance_matrix):
    omega = 0.5  # Inertia weight
    phi_p = 0.2  # Cognitive weight
    phi_g = 0.3  # Social weight

    particles = [Particle(num_customers, num_vehicles, depot, distance_matrix) for _ in range(num_particles)]
    gbest_position = np.copy(particles[0].position)
    gbest_fitness = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            particle.update_velocity(gbest_position, omega, phi_p, phi_g)
            particle.update_position()
            particle.evaluate_fitness()

            if particle.pbest_fitness < gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_position = np.copy(particle.pbest_position)

    return gbest_position, gbest_fitness


# Example usage
num_particles = 10
num_iterations = 100
num_customers = 10
num_vehicles = 3
depot = 1
distance_matrix = [
    [0, 3, 4, 2, 7, 3, 6, 8, 5, 9],
    [3, 0, 5, 4, 6, 8, 7, 2, 9, 5],
    [4, 5, 0, 6, 2, 9, 8, 3, 7, 4],
    [2, 4, 6, 0, 5, 7, 9, 3, 2, 8],
    [7, 6, 2, 5, 0, 4, 6, 3, 2, 7],
    [3, 8, 9, 7, 4, 0, 2, 5, 6, 3],
    [6, 7, 8, 9, 6, 2, 0, 7, 5, 2],
    [8, 2, 3, 3, 3, 5, 7, 0, 2, 6],
    [5, 9, 7, 2, 2, 6, 5, 2, 0, 8],
    [9, 5, 4, 8, 7, 3, 2, 6, 8, 0]
]

gbest_position, gbest_fitness = cooperative_pso(num_particles, num_iterations, num_customers, num_vehicles, depot, distance_matrix)

print("Best position:", gbest_position)
print("Best fitness:", gbest_fitness)