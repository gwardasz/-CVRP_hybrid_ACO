import numpy as np
import random

class Ant:
    def __init__(self, dist_matrix, demands, capacity, alpha=1.0, beta=2.0):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.n_cities = len(demands)
        self.tour = []
        self.total_cost = 0.0
        
    def construct_route(self, pheromones):
        """
        Builds a full CVRP solution (multiple routes).
        """
        # 1. Initialization
        self.tour = [0]  # Start at Depot (Index 0)
        visited = {0}    # Set of visited cities
        current_node = 0
        current_load = 0
        
        # We need to visit all cities (1 to N-1)
        # Assuming demands[0] is 0 (depot has no demand)
        unvisited_cities = set(range(1, self.n_cities))
        
        while unvisited_cities:
            # 2. Identify Feasible Candidates
            # A candidate is feasible if the truck has enough remaining capacity.
            feasible = []
            for city in unvisited_cities:
                if current_load + self.demands[city] <= self.capacity:
                    feasible.append(city)
            
            # 3. Decision Logic
            if not feasible:
                # CONSTRAINT: If no city fits, return to depot to reload
                if current_node != 0:
                    self.tour.append(0)
                    self.total_cost += self.dist_matrix[current_node][0]
                    current_node = 0
                    current_load = 0
                # After reloading, all unvisited cities become feasible again
                # (assuming max(demand) <= capacity)
                continue

            # 4. Selection (Roulette Wheel)
            next_node = self._select_next_node(current_node, feasible, pheromones)
            
            # 5. Move Ant
            self.tour.append(next_node)
            self.total_cost += self.dist_matrix[current_node][next_node]
            current_load += self.demands[next_node]
            visited.add(next_node)
            unvisited_cities.remove(next_node)
            current_node = next_node

        # 6. Final Return to Depot
        if current_node != 0:
            self.tour.append(0)
            self.total_cost += self.dist_matrix[current_node][0]

    def _select_next_node(self, current_node, candidates, pheromones):
        """
        Calculates probabilities and selects the next city using the
        standard ACO transition rule.
        """
        probabilities = []
        
        for city in candidates:
            # Pheromone trail (tau)
            tau = pheromones[current_node][city]
            
            # Heuristic information (eta) = 1 / distance
            # Add small epsilon to avoid division by zero if distance is 0
            dist = self.dist_matrix[current_node][city]
            eta = 1.0 / (dist + 1e-10) 
            
            # Calculate numerator of transition formula
            prob = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append(prob)
        
        # Normalize to get actual probabilities
        prob_sum = sum(probabilities)
        if prob_sum == 0:
            # Fallback if numerators are zero (rare): random choice
            return random.choice(candidates)
            
        normalized_probs = [p / prob_sum for p in probabilities]
        
        # Spin the roulette wheel
        return random.choices(candidates, weights=normalized_probs, k=1)[0]