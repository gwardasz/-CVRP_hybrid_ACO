import optuna
import numpy as np
import sys
import os
from tqdm import tqdm  # <--- NEW: Required for progress bar

# --- IMPORTS ---
from cvrp_data import CVRPInstance
from solver import MMASSolver

# =============================================================================
#  SCIENTIFIC CONFIGURATION
# =============================================================================

# 1. TUNING MODE
# Change this to "standard" or "hybrid" to tune the respective solver.
TUNING_MODE =  "hybrid"
#TUNING_MODE =  "standard"

# 2. TUNING SUITE (The "Mirror" Strategy)
# We use these proxies to ensure parameters work on ALL topologies.
TUNING_INSTANCES = {
    "random":    {"filename": "A-n32-k5.vrp", "bks": 784},   # Small/Random
    "clustered": {"filename": "P-n55-k7.vrp", "bks": 568},   # Clustered/Medium
    "large":     {"filename": "A-n60-k9.vrp", "bks": 1354}   # Larger Scale
}

# 3. OPTUNA SETTINGS
N_TRIALS = 50

# --- ADAPTIVE SETTINGS BASED ON MODE ---
if TUNING_MODE == "hybrid":
    # Hybrid is computationally expensive (Python VND), so we reduce workload.
    # We only need to see if the colony "learns" (Lamarckian), which happens early.
    REPEATS_PER_TRIAL = 5   # Reduced from 10/20
    MAX_ITERATIONS = 20     # Reduced from 100 (Critical fix)
else:
    # Standard mode is fast (Numba), so we can afford rigorous stats.
    REPEATS_PER_TRIAL = 10 
    MAX_ITERATIONS = 100

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def load_tuning_suite():
    """Loads all proxy instances into memory once to save time."""
    suite = []
    print(f"\n[Loader] Loading Tuning Suite for {TUNING_MODE.upper()} mode...")
    
    for key, info in TUNING_INSTANCES.items():
        fname = "./datasets/" + info["filename"]
        if not os.path.exists(fname):
            print(f"  [ERROR] File not found: {fname}")
            print(f"  Please download it from VRPLIB (Augerat Set A/P).")
            sys.exit(1)
        
        instance = CVRPInstance(fname)
        suite.append({
            "type": key,
            "data": instance,
            "bks": info["bks"]
        })
        print(f"  [OK] Loaded {key}: {fname} (n={instance.n_locations})")
    
    return suite

# Global load to allow Optuna to pickle data to workers
loaded_suite = load_tuning_suite()

# =============================================================================
#  OPTIMIZATION OBJECTIVE
# =============================================================================

def objective(trial):
    """
    Minimizes the Weighted Average Percentage Gap to BKS.
    """
    
    # --- 1. HYPERPARAMETER SEARCH SPACE ---
    
    # A. Alpha (Pheromone Importance)
    # Scientific Consensus: Keep fixed at 1.0 to reduce search noise [Stutzle 2000]
    alpha = 1.0 
    
    # B. Beta (Heuristic Importance)
    if TUNING_MODE == "hybrid":
        # Hybrid needs LESS beta (1-3) because Local Search fixes greedy errors
        beta = trial.suggest_float("beta", 1.0, 3.5)
    else:
        # Standard needs MORE beta (2-5) to guide ants correctly
        beta = trial.suggest_float("beta", 2.0, 5.0)

    # C. Rho (Evaporation Rate)
    if TUNING_MODE == "hybrid":
        # Hybrid needs HIGH evaporation (0.1 - 0.5) to forget Lamarckian "super-trails"
        rho = trial.suggest_float("rho", 0.1, 0.5)
    else:
        # Standard needs LOW evaporation (0.01 - 0.1) to slowly build convergence
        rho = trial.suggest_float("rho", 0.01, 0.1)

    # D. Ant Factor (Scalability)
    # Instead of "50 ants", we tune "0.5 * n_cities" vs "1.0 * n_cities"
    ant_factor = trial.suggest_float("ant_factor", 0.5, 1.0, step=0.1)

    # --- 2. EVALUATION LOOP ---
    total_weighted_gap = 0.0
    
    for item in loaded_suite:
        instance = item["data"]
        bks = item["bks"]
        itype = item["type"]
        
        # Calculate dynamic ant count
        n_ants = max(10, int(instance.n_locations * ant_factor))
        
        # Run Repeats
        avg_cost = 0.0
        for _ in range(REPEATS_PER_TRIAL):
            solver = MMASSolver(
                instance,
                n_ants=n_ants,
                rho=rho,
                alpha=alpha,
                beta=beta,
                use_local_search=(TUNING_MODE == "hybrid")
            )
            # CRITICAL: verbose=False prevents parallel print chaos
            cost, _ = solver.solve(max_iterations=MAX_ITERATIONS, verbose=False)
            avg_cost += cost
            
        avg_cost /= REPEATS_PER_TRIAL
        
        # Calculate Gap
        gap = ((avg_cost - bks) / bks) * 100.0
        
        # WEIGHTING STRATEGY
        # We penalize failure on "Clustered" and "Large" maps more than small ones
        if itype == "random":
            weight = 0.2
        elif itype == "clustered":
            weight = 0.4
        else: # large
            weight = 0.4
            
        total_weighted_gap += gap * weight

    return total_weighted_gap

# =============================================================================
#  MAIN EXECUTION (EXTENDED)
# =============================================================================

if __name__ == "__main__":
    # 1. Suppress Optuna logging to avoid fighting with the progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "="*60)
    print(f" STARTING ROBUST TUNING PROTOCOL")
    print(f" Mode: {TUNING_MODE.upper()}")
    print(f" Repeats: {REPEATS_PER_TRIAL} | Parallel Jobs: ALL (-1)")
    print(f" Objective: Minimize Weighted Gap across {len(loaded_suite)} instance types")
    print("="*60)
    print(" Executing Trials (Progress Bar monitors completed trials)...")
    
    study = optuna.create_study(direction="minimize")
    
    # 2. EXTENSION: TQDM Callback for Parallel Progress
    # This runs in the main process and updates as workers finish trials.
    with tqdm(total=N_TRIALS, desc="Hyperparameter Tuning", unit="trial") as pbar:
        def progress_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best Gap": f"{study.best_value:.2f}%"})

        try:
            # n_jobs=-1 uses all CPU cores
            # callbacks=[progress_callback] links the bar to the parallel execution
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, callbacks=[progress_callback])
        except KeyboardInterrupt:
            print("\n[STOP] User interrupted tuning. Saving best so far...")

    print("\n" + "#"*60)
    print(" TUNING COMPLETE")
    print("#"*60)
    print(f"Best Weighted Gap: {study.best_value:.2f}%")
    print("Recommended Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key} = {value}")
    
    print("\n[COPY TO MAIN.PY]")
    print(f"PARAMS_{TUNING_MODE.upper()} = {{")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value:.4f},")
    print(f"    'alpha': 1.0  # Fixed")
    print("}")