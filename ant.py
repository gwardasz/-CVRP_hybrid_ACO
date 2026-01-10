# ant.py
import numpy as np
from fast_ant import fast_construct_route, calculate_cost

class Ant:
    def __init__(self, dist_matrix, demands, capacity, alpha=1.0, beta=2.0):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tour = []
        self.total_cost = 0.0

    def construct_route(self, pheromones):
        # Pass numpy arrays to the JIT function
        # Ensure pheromones is a numpy array (float64)
        self.tour = fast_construct_route(
            self.dist_matrix, 
            self.demands, 
            self.capacity, 
            pheromones, 
            self.alpha, 
            self.beta
        )
        
        # Calculate cost (Numba can also speed this up, but it's fast enough)
        self.total_cost = calculate_cost(self.tour, self.dist_matrix)