import numpy as np
import random
from numba import njit

@njit
def fast_construct_route(dist_matrix, demands, capacity, pheromones, alpha, beta, start_node=0):
    n_cities = len(demands)
    visited = np.zeros(n_cities, dtype=np.bool_) 
    tour = np.zeros(n_cities * 2, dtype=np.int32) # Fixed size array
    
    current_node = start_node
    current_load = 0
    visited[start_node] = True
    
    tour_idx = 0
    tour[tour_idx] = start_node
    tour_idx += 1
    
    cities_visited = 1
    
    while cities_visited < n_cities:
        # --- PASS 1: Calculate Sum of Probabilities ---
        prob_sum = 0.0
        # We don't store individual probs, just the sum
        # We need a way to know if ANY move is possible
        move_possible = False
        
        for city in range(n_cities):
            if not visited[city]:
                if current_load + demands[city] <= capacity:
                    move_possible = True
                    # Calculate Prob Terms
                    tau = pheromones[current_node, city]
                    # Direct access is faster
                    d = dist_matrix[current_node, city]
                    eta = 1.0 / (d + 1e-10)
                    
                    # Accumulate Sum
                    prob_sum += (tau ** alpha) * (eta ** beta)
        
        # --- DECISION LOGIC ---
        next_node = -1
        
        if not move_possible:
            # Must return to depot
            if current_node != 0:
                next_node = 0
                current_load = 0 # Refill
                # Do NOT mark depot as visited (it's always available)
            else:
                # Stuck at depot with unvisited cities? Should not happen in feasible VRP
                break 
        else:
            # --- PASS 2: Roulette Wheel Selection ---
            # Pick a random threshold
            pick = random.random() * prob_sum
            current_sum = 0.0
            
            # Re-iterate to find the winner
            # This is slightly redundant calculation but saves O(N) memory allocation
            # which is worth it in JIT compilation
            for city in range(n_cities):
                if not visited[city]:
                    if current_load + demands[city] <= capacity:
                        tau = pheromones[current_node, city]
                        d = dist_matrix[current_node, city]
                        eta = 1.0 / (d + 1e-10)
                        p = (tau ** alpha) * (eta ** beta)
                        
                        current_sum += p
                        if current_sum >= pick:
                            next_node = city
                            break
            
            # Fallback for floating point errors
            if next_node == -1:
                # Just pick the last feasible one
                for city in range(n_cities - 1, -1, -1):
                     if not visited[city] and (current_load + demands[city] <= capacity):
                         next_node = city
                         break
                         
        # --- UPDATE STATE ---
        tour[tour_idx] = next_node
        tour_idx += 1
        current_node = next_node
        
        if next_node != 0:
            visited[next_node] = True
            cities_visited += 1
            current_load += demands[next_node]
        else:
            # Returned to depot
            current_load = 0

    # Final return to depot if not already there
    if current_node != 0:
        tour[tour_idx] = 0
        tour_idx += 1

    return tour[:tour_idx]

@njit
def calculate_cost(tour, dist_matrix):
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i], tour[i+1]]
    return cost