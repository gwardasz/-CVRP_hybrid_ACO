import numpy as np
from ant import Ant

class ASSolver:
    def __init__(self, problem_instance, n_ants=20, rho=0.5, alpha=1.0, beta=2.5):
        """
        Initializes the Standard Ant System (AS) Solver.
        Note: AS usually requires higher evaporation (rho ~0.5) than MMAS (rho ~0.02)
        because ALL ants add pheromones, which accumulates faster.
        """
        self.dist_matrix, self.demands, self.capacity = problem_instance.get_data()
        self.n_cities = len(self.demands)
        
        # Hyperparameters
        self.n_ants = n_ants
        self.rho = rho        # Evaporation rate
        self.alpha = alpha    # Pheromone importance
        self.beta = beta      # Heuristic importance
        
        # --- DIFFERENCE 1: Initialization ---
        # AS starts with small/neutral pheromones, not Max.
        # We use a small constant (e.g., 1.0 or 0.1).
        self.pheromones = np.ones((self.n_cities, self.n_cities)) * 0.1
        
        # Global best tracking
        self.best_global_solution = []
        self.best_global_cost = float('inf')

    def solve(self, max_iterations=100):
        print(f"Starting Ant System (AS) Optimization for {max_iterations} iterations...")
        
        for iteration in range(max_iterations):
            # 1. Construction Phase
            ants = []
            for _ in range(self.n_ants):
                ant = Ant(self.dist_matrix, self.demands, self.capacity, 
                          self.alpha, self.beta)
                ant.construct_route(self.pheromones)
                ants.append(ant)
            
            # 2. Find Iteration Best (Just for tracking history/logs)
            ants.sort(key=lambda x: x.total_cost)
            best_ant_iter = ants[0]
            
            # 3. Update Global Best
            if best_ant_iter.total_cost < self.best_global_cost:
                self.best_global_cost = best_ant_iter.total_cost
                self.best_global_solution = best_ant_iter.tour
                print(f"Iter {iteration+1}: New Best Found! Cost = {self.best_global_cost:.2f}")

            # --- DIFFERENCE 2: Pheromone Update ---
            # In AS, we pass the entire list of 'ants', not just the best one.
            self._update_pheromones(ants)
            
        return self.best_global_cost, self.best_global_solution

    def _update_pheromones(self, ants):
        """
        Standard AS Update Rule:
        1. Evaporate everyone.
        2. ALL ants deposit pheromones.
        3. NO Limits/Clamping.
        """
        # A. Evaporation
        self.pheromones *= (1.0 - self.rho)
        
        # B. All Ants Deposit
        for ant in ants:
            # Calculate deposit amount (Q / L)
            deposit = 1.0 / ant.total_cost
            
            tour = ant.tour
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                # Symmetric Update
                self.pheromones[u][v] += deposit
                self.pheromones[v][u] += deposit
        
        # --- DIFFERENCE 3: No Clamping ---
        # In AS, we do NOT force values between min/max.