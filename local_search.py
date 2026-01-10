import numpy as np

class LocalSearch:
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix

    def optimize_solution(self, full_tour):
        """
        Parses the full ACO tour (with multiple 0s), optimizes each sub-route
        using 2-Opt, and returns the improved full tour and new cost.
        """
        # 1. Split full tour into individual vehicle routes
        routes = self._extract_routes(full_tour)
        
        improved_routes = []
        total_cost = 0.0
        
        # 2. Apply 2-Opt to each route independently
        for route in routes:
            optimized_route = self._two_opt(route)
            improved_routes.append(optimized_route)
            
            # Calculate cost for this segment
            # Ensure we account for return to depot distance
            route_cost = self._calculate_route_cost(optimized_route)
            total_cost += route_cost

        # 3. Reconstruct the full flattened tour
        # Start with Depot (0)
        final_tour = [0]
        for r in improved_routes:
            # Append route elements + return to depot
            final_tour.extend(r)
            final_tour.append(0)
            
        return final_tour, total_cost

    def _two_opt(self, route):
        """
        Standard Best-Improvement 2-Opt for a single route.
        Route input format: [1, 5, 2] (No depots included in this list)
        """
        best_route = route[:]
        improved = True
        
        # We need the depot logic for distance calculation, 
        # so we temporarily prepend/append 0 for the math
        # but we only swap the inner customers.
        
        while improved:
            improved = False
            # Check every pair of edges (i, i+1) and (j, j+1)
            # Route indices: 0 to len(route)
            n = len(best_route)
            
            # We treat the route as 0 -> node -> ... -> node -> 0
            # To simplify, let's work with the raw indices of the customer list
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if j - i == 1: continue # Adjacent edges, no change
                    
                    # Calculate cost change if we reverse segment [i+1...j]
                    if self._check_improvement(best_route, i, j):
                        # Perform the swap (reverse the segment)
                        best_route[i+1 : j+1] = best_route[i+1 : j+1][::-1]
                        improved = True
        
        return best_route

    def _check_improvement(self, route, i, j):
        """
        Checks if swapping edges (i, i+1) and (j, j+1) reduces length.
        Note: We must handle the implicit depot connections at start/end.
        """
        # This helper is simplified. In a real efficient implementation,
        # you calculate delta only for the broken edges.
        
        # Construct current full path for these 4 points
        A = route[i]
        B = route[i+1]
        C = route[j]
        D = route[j+1] if j+1 < len(route) else -1 # -1 implies boundary handling
        
        # For simplicity in this snippets, calculating full path delta is safer
        # but slower. Let's do the standard Delta check:
        # Distance(A, B) + Distance(C, D)  vs  Distance(A, C) + Distance(B, D)
        
        # NOTE: Handling the depot boundaries correctly in 2-opt is tricky. 
        # A robust way is to construct the actual path with 0s for calculation:
        temp_route = [0] + route + [0]
        # Adjust indices because of the prepended 0
        idx_i = i + 1
        idx_j = j + 1
        
        A = temp_route[idx_i]
        B = temp_route[idx_i+1]
        C = temp_route[idx_j]
        D = temp_route[idx_j+1]
        
        current_dist = self.dist_matrix[A][B] + self.dist_matrix[C][D]
        new_dist = self.dist_matrix[A][C] + self.dist_matrix[B][D]
        
        return new_dist < current_dist

    def _extract_routes(self, tour):
        """Splits [0, 1, 2, 0, 3, 4, 0] into [[1,2], [3,4]]"""
        routes = []
        current_route = []
        for node in tour:
            if node == 0:
                if current_route:
                    routes.append(current_route)
                    current_route = []
            else:
                current_route.append(node)
        if current_route:
            routes.append(current_route)
        return routes

    def _calculate_route_cost(self, route):
        cost = 0
        prev = 0 # Start at depot
        for node in route:
            cost += self.dist_matrix[prev][node]
            prev = node
        cost += self.dist_matrix[prev][0] # Return to depot
        return cost