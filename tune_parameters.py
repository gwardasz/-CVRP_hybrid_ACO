import optuna
import numpy as np
import sys
import time

# --- IMPORTS ---
from cvrp_data import CVRPInstance
from solver import MMASSolver

# --- CONFIGURATION ---
INSTANCE_FILENAME = "E-n51-k5.vrp"

# OPTUNA SETTINGS
N_TRIALS = 50           # Total parameter combinations to test per study
REPEATS_PER_TRIAL = 20  # How many times to run EACH combination
TUNING_ITERATIONS = 100 # Run shorter simulations for tuning

# --- LOAD DATA (Global) ---
print(f"Loading dataset: {INSTANCE_FILENAME}...")
try:
    GLOBAL_INSTANCE = CVRPInstance(INSTANCE_FILENAME)
    print(f"Successfully loaded {GLOBAL_INSTANCE.name} with {GLOBAL_INSTANCE.n_locations} nodes.")
except FileNotFoundError:
    print(f"ERROR: Could not find file '{INSTANCE_FILENAME}'.")
    sys.exit(1)

def run_study(fixed_n_ants):
    """
    Runs a complete Optuna study for a fixed number of ants.
    """
    
    def objective(trial):
        """
        Optimization Objective: Minimize the MEAN cost over 20 stochastic runs.
        """
        # 1. Suggest Hyperparameters
        # Note: n_ants is NOT suggested here, it is fixed from the parent function
        
        # Alpha (Pheromone Importance): Range [0.5, 5.0]
        alpha = trial.suggest_float("alpha", 0.5, 5.0)      

        # Beta (Heuristic Importance): Range [1.0, 6.0]
        beta = trial.suggest_float("beta", 1.0, 6.0)        

        # Rho (Evaporation Rate): Range [0.005, 0.2] (Low for MMAS)
        rho = trial.suggest_float("rho", 0.005, 0.2)         

        # 2. Evaluation Loop (Stochastic Averaging)
        costs = []
        
        for i in range(REPEATS_PER_TRIAL):
            solver = MMASSolver(
                GLOBAL_INSTANCE,
                n_ants=fixed_n_ants, # USE THE FIXED VALUE
                rho=rho,
                alpha=alpha,
                beta=beta
            )
            
            # Run solver (Pruning disabled)
            final_cost, _ = solver.solve(max_iterations=TUNING_ITERATIONS)
            costs.append(final_cost)

        # 3. Return the Mean Cost
        return np.mean(costs)

    # --- Run the Study ---
    print("\n" + "="*60)
    print(f" STARTING OPTIMIZATION FOR N_ANTS = {fixed_n_ants}")
    print("="*60)
    
    study = optuna.create_study(direction="minimize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    except KeyboardInterrupt:
        print("\nTuning interrupted! Saving current best...")

    return study

if __name__ == "__main__":
    start_time = time.time()
    
    # --- STUDY 1: 25 ANTS ---
    study_25 = run_study(25)
    
    # --- STUDY 2: 50 ANTS ---
    study_50 = run_study(50)

    # --- FINAL REPORT ---
    elapsed = time.time() - start_time
    print("\n" + "#"*60)
    print(" ALL TUNING COMPLETE")
    print("#"*60)
    print(f"Total Time: {elapsed/60:.2f} minutes")

    print("\n--- RESULTS FOR 25 ANTS ---")
    print(f"Best Mean Cost: {study_25.best_value:.2f}")
    print("Recommended Parameters:")
    for key, value in study_25.best_params.items():
        print(f"  {key} = {value}")
    print(f"  n_ants = 25 (Fixed)")

    print("\n--- RESULTS FOR 50 ANTS ---")
    print(f"Best Mean Cost: {study_50.best_value:.2f}")
    print("Recommended Parameters:")
    for key, value in study_50.best_params.items():
        print(f"  {key} = {value}")
    print(f"  n_ants = 50 (Fixed)")