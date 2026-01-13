import optuna
import numpy as np
import sys
import os
import argparse

# --- IMPORTS ---
from cvrp_data import CVRPInstance
from solver import MMASSolver

# =============================================================================
#  COMMAND LINE ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Scientific Parameter Tuning for MMAS")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["standard", "hybrid"], 
        default="hybrid",
        help="Select tuning mode: 'standard' (Pure MMAS) or 'hybrid' (MMAS + VND)"
    )
    return parser.parse_args()

args = parse_arguments()
TUNING_MODE = args.mode

# =============================================================================
#  CONFIGURATION
# =============================================================================

# 1. FILE SYSTEM
DATASET_DIR = "./datasets"

# 2. CALIBRATION SUITE
CALIBRATION_INSTANCES = [
    {"file": "A-n32-k5.vrp", "bks": 784},   # Small
    {"file": "P-n55-k7.vrp", "bks": 568},   # Medium
    {"file": "A-n60-k9.vrp", "bks": 1354}   # Large
]

# 3. COMPUTATIONAL BUDGET
N_TRIALS = 50

if TUNING_MODE == "hybrid":
    REPEATS_PER_TRIAL = 6   
    MAX_ITERATIONS = 15     
else:
    REPEATS_PER_TRIAL = 6 
    MAX_ITERATIONS = 150

# =============================================================================
#  DATA LOADING
# =============================================================================

def load_calibration_set():
    suite = []
    print(f"\n[Loader] Loading Calibration Suite for {TUNING_MODE.upper()} mode...")
    
    for item in CALIBRATION_INSTANCES:
        fname = os.path.join(DATASET_DIR, item["file"])
        if not os.path.exists(fname):
            print(f"  [ERROR] File not found: {fname}")
            sys.exit(1)
        
        try:
            instance = CVRPInstance(fname)
            suite.append({"data": instance, "bks": item["bks"]})
            print(f"  [OK] Loaded: {instance.name} (n={instance.n_locations})")
        except Exception as e:
            print(f"  [ERROR] Failed to parse {fname}: {e}")
            sys.exit(1)
    return suite

calibration_suite = load_calibration_set()

# =============================================================================
#  OBJECTIVE FUNCTION
# =============================================================================

def objective(trial):
    """
    Minimizes the Mean Percentage Gap (ARPD).
    """
    
    # --- 1. PARAMETER SEARCH SPACE (PHYSICS ONLY) ---
    
    # A. Alpha (Fixed)
    alpha = 1.0 
    
    # B. Beta (Heuristic Importance) - Range [1.0, 5.0]
    beta = trial.suggest_float("beta", 1.0, 5.0)

    # C. Rho (Evaporation) - Range [0.01, 0.5]
    rho = trial.suggest_float("rho", 0.01, 0.5)

    # D. Ants (FIXED to N per your request)
    # We do NOT tune this. We just use n_ants = instance_size.
    
    # --- 2. EVALUATION LOOP ---
    total_gap = 0.0
    
    for item in calibration_suite:
        instance = item["data"]
        bks = item["bks"]
        
        # FIXED: Ants = Number of Nodes (m=N)
        n_ants = instance.n_locations
        
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
            
            # Robust extraction of cost (handles 2 or 3 return values)
            result = solver.solve(max_iterations=MAX_ITERATIONS, verbose=False)
            
            if isinstance(result, (tuple, list)):
                cost = result[0]
            else:
                cost = result
                
            avg_cost += cost
            
        avg_cost /= REPEATS_PER_TRIAL
        gap = ((avg_cost - bks) / bks) * 100.0
        total_gap += gap

    # --- 3. SCORING (ARPD) ---
    mean_gap = total_gap / len(calibration_suite)

    return mean_gap

# =============================================================================
#  MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(f" STARTING SCIENTIFIC TUNING PROTOCOL")
    print(f" Mode:    {TUNING_MODE.upper()}")
    print(f" Ants:    Fixed to Problem Size (m=N)")
    print(f" Budget:  {REPEATS_PER_TRIAL} repeats x {MAX_ITERATIONS} iterations")
    print("="*60)
    
    study = optuna.create_study(direction="minimize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted tuning.")

    print("\n" + "#"*60)
    print(" TUNING COMPLETE")
    print("#"*60)
    print(f"Best Mean Gap (ARPD): {study.best_value:.2f}%")
    print("Recommended Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key} = {value}")
    
    print("\n[COPY TO MAIN.PY]")
    print(f"PARAMS_{TUNING_MODE.upper()} = {{")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value:.4f},")
    print(f"    'alpha': 1.0,  # Fixed")
    # Note: We recommend you use dynamic 'n_ants' logic in main.py, 
    # but for these params, the physics (rho/beta) are now optimized.
    print("}")