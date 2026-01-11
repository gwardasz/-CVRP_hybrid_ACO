try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
import numpy as np
from ant import Ant
from local_search import LocalSearch


class MMASSolver:
    def __init__(self, problem_instance, n_ants=50, rho=0.2, alpha=1.0, beta=2.8, use_local_search=True):
        """
        Initializes the Max-Min Ant System Solver.
        """
        # Unpack problem data
        self.dist_matrix, self.demands, self.capacity = problem_instance.get_data()
        self.n_cities = len(self.demands)
        
        # Hyperparameters
        self.n_ants = n_ants
        self.rho = rho       # Evaporation rate
        self.alpha = alpha   # Pheromone importance
        self.beta = beta     # Heuristic importance
        self.use_local_search = use_local_search
        
        # MMAS Specific Parameters [cite: 12]
        # These will be dynamically adjusted or set based on the first solution
        self.tau_max = 1.0  
        self.tau_min = 0.001 
        
        # Initialize Pheromones to tau_max (optimistic initialization) [cite: 12]
        self.pheromones = np.full((self.n_cities, self.n_cities), self.tau_max)
        
        # Global best tracking
        self.best_global_solution = []
        self.best_global_cost = float('inf')

    def solve(self, max_iterations=100, verbose=True):
        #print(f"Starting MMAS Optimization for {max_iterations} iterations... ({self.n_ants} ants)")
        
        # Initialize Local Search engine
        ls_engine = LocalSearch(self.dist_matrix, self.demands, self.capacity)

        # 1. Setup Progress Bar (only if verbose=True and tqdm is installed)
        if verbose and HAS_TQDM:
            iterator = tqdm(range(max_iterations), desc="MMAS Progress", unit="iter")
        else:
            iterator = range(max_iterations)

        for iteration in iterator:
            # 1. Construction Phase
            ants = []
            for _ in range(self.n_ants):
                ant = Ant(self.dist_matrix, self.demands, self.capacity, 
                          self.alpha, self.beta)
                ant.construct_route(self.pheromones)                
                ants.append(ant)
            
            # 2. Sort to find the "Raw" Best Ant
            ants.sort(key=lambda x: x.total_cost)
            best_ant_iter = ants[0]

            # --- OPTIMIZATION: ELITIST STRATEGY ---
            # Only run heavy Local Search on the winner!
            if self.use_local_search:
                improved_tour, improved_cost = ls_engine.optimize_solution(best_ant_iter.tour)
                
                # Update the ant with the optimized result
                best_ant_iter.tour = np.array(improved_tour, dtype=np.int32)
                best_ant_iter.total_cost = improved_cost
            # ---------------------------------------
            
            # 3. Update Global Best (Check if the optimized ant beats the record)
            if best_ant_iter.total_cost < self.best_global_cost:
                self.best_global_cost = best_ant_iter.total_cost
                self.best_global_solution = best_ant_iter.tour.copy()
                
                #print(f"Iter {iteration+1}: New Best Found! Cost = {self.best_global_cost:.2f}")
                
                # Dynamic MMAS Limits
                self._update_tau_limits()
                # UPDATE PROGRESS BAR with new best score
                if verbose and HAS_TQDM:
                    iterator.set_postfix({"Best Cost": f"{self.best_global_cost:.2f}"})
                
            # 4. Pheromone Update (MMAS Logic)
            self._update_pheromones(best_ant_iter)
            
        return self.best_global_cost, self.best_global_solution

    def _update_tau_limits(self):
        """
        Recalculates tau_max and tau_min based on the new best cost found.
        Uses the Stützle & Hoos (2000) derivation.
        """
        # 1. Update tau_max: 1 / (rho * best_cost)
        if self.best_global_cost > 0:
            self.tau_max = 1.0 / (self.rho * self.best_global_cost)
        else:
            self.tau_max = 1.0  # Fallback if cost is 0 or uninitialized

        # 2. Update tau_min
        # n is the number of customers (decisions to make)
        n = len(self.demands) - 1 
        
        # p_best is the probability of constructing the best tour (0.05 is standard)
        p_best = 0.05
        
        # avg is the average number of choices (n/2 is standard approx)
        avg = n / 2.0

        if n > 1:
            # The Formula:
            # tau_min = tau_max * (1 - p_best^(1/n)) / ((avg - 1) * p_best^(1/n))
            root_p = p_best ** (1.0 / n)
            numerator = self.tau_max * (1.0 - root_p)
            denominator = (avg - 1.0) * root_p
            
            self.tau_min = numerator / denominator
        else:
            # Fallback for very small/trivial instances
            self.tau_min = self.tau_max / 20.0

        # 3. Safety Check: Ensure min never exceeds max
        if self.tau_min > self.tau_max:
            self.tau_min = self.tau_max


    def _update_pheromones(self, best_ant):
        """
        Applies Evaporation and Elitist Deposit, then Clamps values.
        """
        # A. Evaporation: Decrease all pheromones by factor rho
        self.pheromones *= (1.0 - self.rho)
        
        # B. Elitist Deposit: Only the BEST ant deposits pheromones 
        # Calculate deposit amount (1 / Cost)
        deposit = 1.0 / best_ant.total_cost
        
        # Walk through the tour and add pheromones to used edges
        tour = best_ant.tour
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            # Symmetric TSP/VRP: Update both directions
            self.pheromones[u][v] += deposit
            self.pheromones[v][u] += deposit
            
        # C. Stagnation Control: Clamp between tau_min and tau_max 
        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)