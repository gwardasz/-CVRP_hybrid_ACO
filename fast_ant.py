import numpy as np
import random
from numba import njit

@njit  # <--- This is the magic line
def fast_construct_route(dist_matrix, demands, capacity, pheromones, alpha, beta, start_node=0):
    """
    Stand-alone function compiled to Machine Code.
    Does NOT use 'self'. Everything is passed as numpy arrays.
    """
    n_cities = len(demands)
    visited = np.zeros(n_cities, dtype=np.bool_) # Numba prefers arrays over sets
    tour = np.empty(n_cities * 2, dtype=np.int32) # Pre-allocate max size (safe buffer)
    
    # Initialize
    current_node = start_node
    current_load = 0
    visited[start_node] = True
    tour_idx = 0
    tour[tour_idx] = start_node
    tour_idx += 1
    
    # We need to visit n_cities - 1 customers
    cities_visited_count = 1 
    
    while cities_visited_count < n_cities:
        # 1. Identify Feasible Candidates
        # In Numba, manual loops are often faster than list comprehensions
        candidates = []
        probabilities = []
        prob_sum = 0.0
        
        for city in range(n_cities):
            if not visited[city]:
                if current_load + demands[city] <= capacity:
                    # Calculate Probability Numerator directly here for speed
                    tau = pheromones[current_node, city]
                    dist = dist_matrix[current_node, city]
                    eta = 1.0 / (dist + 1e-10)
                    prob = (tau ** alpha) * (eta ** beta)
                    
                    candidates.append(city)
                    probabilities.append(prob)
                    prob_sum += prob
        
        # 2. Decision Logic
        if len(candidates) == 0:
            # No feasible city? Go back to depot.
            if current_node != 0:
                tour[tour_idx] = 0
                tour_idx += 1
                current_node = 0
                current_load = 0
                # Don't increment cities_visited_count, we just reloaded
                continue
            else:
                # If we are at depot and can't move, something is wrong (or done)
                break

        # 3. Roulette Wheel Selection (Manual implementation for Numba)
        # Numba supports random.random(), but not random.choices with weights
        pick = random.random() * prob_sum
        current = 0.0
        next_node = candidates[-1] # Default fallback
        
        for i in range(len(candidates)):
            current += probabilities[i]
            if current >= pick:
                next_node = candidates[i]
                break
        
        # 4. Move
        tour[tour_idx] = next_node
        tour_idx += 1
        current_load += demands[next_node]
        visited[next_node] = True
        cities_visited_count += 1
        current_node = next_node

    # 5. Final Return to Depot
    if current_node != 0:
        tour[tour_idx] = 0
        tour_idx += 1
        
    # Return valid slice of the array
    return tour[:tour_idx]

@njit
def calculate_cost(tour, dist_matrix):
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i], tour[i+1]]
    return cost