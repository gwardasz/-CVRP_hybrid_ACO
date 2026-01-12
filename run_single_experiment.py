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

# Scientific Parameters (Matched to test.py)
PARAMS_STANDARD = {
    'alpha': 1.0, 
    'beta': 4.67, 
    'rho': 0.10, 
    'ant_factor': 1.0
}

PARAMS_HYBRID = {
    'alpha': 1.0, 
    # Reverted to shorter precision as requested to match 'test.py'
    'beta': 3.24,   
    'rho': 0.22, 
    'ant_factor': 1.0
}

N_TRIALS = 20      
MAX_ITERS = 200    
RESULTS_DIR = "results"

# =============================================================================
#  CORE RUNNER
# =============================================================================

def run_trial(instance, params, use_ls):
    n_ants = max(10, int(round(instance.n_locations * params['ant_factor'])))
    solver = MMASSolver(
        instance, n_ants=n_ants, rho=params['rho'], 
        alpha=params['alpha'], beta=params['beta'], use_local_search=use_ls
    )
    
    t0 = time.time()
    cost, _, history = solver.solve(max_iterations=MAX_ITERS, verbose=False)
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
    convergence_data = {}

    # 3. Run Standard MMAS
    desc_std = f"Std {filename.split('.')[0]}"
    for i in tqdm(range(N_TRIALS), desc=desc_std, ncols=80):
        c, t, h = run_trial(instance, PARAMS_STANDARD, use_ls=False)
        detailed_data.append({
            'Instance': filename.replace('.vrp', ''),
            'Size': instance.n_locations, # Critical for Scalability Plot
            'Algorithm': 'Standard',
            'Cost': c, 
            'Time': t, 
            'BKS': real_bks
        })
        # Save first trial convergence for plotting
        if i == 0: convergence_data['Standard'] = h

    # 4. Run Hybrid MMAS
    desc_hyb = f"Hyb {filename.split('.')[0]}"
    for i in tqdm(range(N_TRIALS), desc=desc_hyb, ncols=80):
        c, t, h = run_trial(instance, PARAMS_HYBRID, use_ls=True)
        detailed_data.append({
            'Instance': filename.replace('.vrp', ''),
            'Size': instance.n_locations,
            'Algorithm': 'Hybrid',
            'Cost': c, 
            'Time': t, 
            'BKS': real_bks
        })
        if i == 0: convergence_data['Hybrid'] = h

    # 5. Save Results to Unique Files
    base_name = filename.replace('.vrp', '')
    csv_path = os.path.join(RESULTS_DIR, f"{base_name}_results.csv")
    json_path = os.path.join(RESULTS_DIR, f"{base_name}_convergence.json")

    pd.DataFrame(detailed_data).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(convergence_data, f)

    print(f"\n[Success] Saved results to: {csv_path}")

if __name__ == "__main__":
    main()