import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm

# Import your project modules
from cvrp_data import CVRPInstance
from solver import MMASSolver

# =============================================================================
#  CONFIGURATION
# =============================================================================

PARAMS_STANDARD = {
    'alpha': 1.0, 
    'beta': 4.18,
    'rho': 0.10,
    'ant_factor': 1.0
}

PARAMS_HYBRID = {
    'alpha': 1.0,
    'beta': 3.13,
    'rho': 0.50,
    'ant_factor': 1.0
}

N_TRIALS = 20      

# SCIENTIFIC ADJUSTMENT:
# Hybrid converges faster but iterations are costlier. 
# We adjust budgets to ensure comparable total runtime windows.
MAX_ITERS_STD = 200
MAX_ITERS_HYB = 25 # 50  

RESULTS_DIR = "results"

# =============================================================================
#  CORE RUNNER
# =============================================================================

def run_trial(instance, params, use_ls, max_iters):
    # Dynamic ant calculation based on problem size
    n_ants = max(10, int(round(instance.n_locations * params['ant_factor'])))
    
    solver = MMASSolver(
        instance, n_ants=n_ants, rho=params['rho'], 
        alpha=params['alpha'], beta=params['beta'], use_local_search=use_ls
    )
    
    t0 = time.time()
    # Pass the specific max_iters for this mode
    cost, _, history = solver.solve(max_iterations=max_iters, verbose=False)
    elapsed = time.time() - t0
    
    return cost, elapsed, history

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run ACO Experiment for a single dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Filename (e.g., CMT1.vrp)")
    args = parser.parse_args()

    filename = args.dataset
    file_path = os.path.join("./datasets", filename)
    
    # 2. Validation
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"Running Experiment for: {filename}")
    instance = CVRPInstance(file_path)
    
    # Attempt to load BKS (Best Known Solution) for Gap calculation
    real_bks = instance.bks if (hasattr(instance, 'bks') and instance.bks is not None) else -1.0
    
    detailed_data = []
    # REFACTOR: Initialize lists to hold history for ALL trials
    convergence_data = {
        'Standard': [],
        'Hybrid': [],
        'Meta_Max_Iters': {'Standard': MAX_ITERS_STD, 'Hybrid': MAX_ITERS_HYB}
    }

    # 3. Run Standard MMAS
    desc_std = f"Std {filename.split('.')[0]}"
    for i in tqdm(range(N_TRIALS), desc=desc_std, ncols=80):
        c, t, h = run_trial(instance, PARAMS_STANDARD, use_ls=False, max_iters=MAX_ITERS_STD)
        detailed_data.append({
            'Instance': filename.replace('.vrp', ''),
            'Size': instance.n_locations, 
            'Algorithm': 'Standard',
            'Cost': c, 
            'Time': t, 
            'BKS': real_bks,
            'Iterations': MAX_ITERS_STD,
            'LocalSearch': False
        })
        # REFACTOR: Save history for every trial
        convergence_data['Standard'].append(h)

    # 4. Run Hybrid MMAS
    desc_hyb = f"Hyb {filename.split('.')[0]}"
    for i in tqdm(range(N_TRIALS), desc=desc_hyb, ncols=80):
        c, t, h = run_trial(instance, PARAMS_HYBRID, use_ls=True, max_iters=MAX_ITERS_HYB)
        detailed_data.append({
            'Instance': filename.replace('.vrp', ''),
            'Size': instance.n_locations,
            'Algorithm': 'Hybrid',
            'Cost': c, 
            'Time': t, 
            'BKS': real_bks,
            'Iterations': MAX_ITERS_HYB,
            'LocalSearch': True
        })
        # REFACTOR: Save history for every trial
        convergence_data['Hybrid'].append(h)

    # 5. Save Results
    base_name = filename.replace('.vrp', '')
    csv_path = os.path.join(RESULTS_DIR, f"{base_name}_results.csv")
    json_path = os.path.join(RESULTS_DIR, f"{base_name}_convergence.json")

    try:
        pd.DataFrame(detailed_data).to_csv(csv_path, index=False)
        print(f"\n[Success] CSV saved to: {csv_path}")
    except Exception as e:
        print(f"\n[Error] Failed to save CSV: {e}")

    try:
        with open(json_path, "w") as f:
            json.dump(convergence_data, f)
        print(f"[Success] JSON saved to: {json_path}")
    except Exception as e:
        print(f"\n[Error] Failed to save JSON: {e}")

if __name__ == "__main__":
    main()