import matplotlib.pyplot as plt
import numpy as np
import time
import os
from cvrp_data import CVRPInstance
from solver import MMASSolver

# =============================================================================
#  SCIENTIFIC PARAMETER SETS (Result of Optuna Tuning)
# =============================================================================

# 1. PARAMETERS FOR PURE MMAS (Standard Mode)
# Tuned for: Low Evaporation, High Heuristic Reliance, Higher Ant Count
PARAMS_STANDARD = {
    'alpha': 1.0,      # Fixed
    'beta': 4.67,      # Higher beta because ants rely on greedy distance
    'rho': 0.10,       # Lower rho to build history slowly
    'ant_factor': 1.0  # Standard density: 1 ant per city
}

# 2. PARAMETERS FOR HYBRID MMAS (VND + Lamarckian)
# Tuned for: High Evaporation, Low Heuristic Reliance, Efficient Ant Count
PARAMS_HYBRID = {
    'beta': 3.2444,
    'rho': 0.2203,
    'ant_factor': 1.0000,
    'alpha': 1.0  # Fixed
}

# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================

def plot_comparison(instance, tour1, cost1, title1, tour2, cost2, title2):
    """
    Plots two solutions side-by-side for visual comparison.
    """
    coords = instance.coords
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    titles = [f"{title1}\nCost: {cost1:.2f}", f"{title2}\nCost: {cost2:.2f}"]
    tours = [tour1, tour2]
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        
        # Plot Nodes
        ax.scatter(coords[0][0], coords[0][1], c='red', marker='s', s=100, label='Depot', zorder=5)
        cust_x = [c[0] for c in coords[1:]]
        cust_y = [c[1] for c in coords[1:]]
        ax.scatter(cust_x, cust_y, c='blue', s=30, alpha=0.6, label='Customers')
        
        # Plot Routes
        tour = tours[i]
        if tour is not None and len(tour) > 0:
            x_seq = [coords[node][0] for node in tour]
            y_seq = [coords[node][1] for node in tour]
            
            # Draw Lines
            ax.plot(x_seq, y_seq, c='black', linewidth=1, alpha=0.7, linestyle='-')
            
            # Draw Arrows (Quiver)
            ax.quiver(x_seq[:-1], y_seq[:-1], 
                      [x_seq[j+1]-x_seq[j] for j in range(len(x_seq)-1)],
                      [y_seq[j+1]-y_seq[j] for j in range(len(y_seq)-1)], 
                      scale_units='xy', angles='xy', scale=1, color='black', width=0.003, headwidth=4)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# =============================================================================
#  CORE EXPERIMENT LOGIC
# =============================================================================

def run_single_experiment(instance, params, use_local_search, max_iters, verbose=True):
    """
    Runs a single MMAS experiment with DYNAMIC SCALING.
    """
    mode = "Hybrid (VND)" if use_local_search else "Pure MMAS"
    
    # --- SCIENTIFIC SCALING: CALCULATE ANTS DYNAMICALLY ---
    # m = Round(ant_factor * N)
    n_ants = int(round(instance.n_locations * params['ant_factor']))
    n_ants = max(10, n_ants) # Safety floor: never less than 10
    
    if verbose:
        print(f"\n>>> Running {mode}...")
        print(f"    [Config] Ants: {n_ants} (Factor: {params['ant_factor']}), Rho: {params['rho']}, Beta: {params['beta']}")
    
    t0 = time.time()
    
    solver = MMASSolver(
        instance, 
        n_ants=n_ants,          # Passed Dynamic Count
        rho=params['rho'], 
        alpha=params['alpha'], 
        beta=params['beta'], 
        use_local_search=use_local_search
    )
    
    cost, tour = solver.solve(max_iterations=max_iters, verbose=verbose)
    elapsed = time.time() - t0
    
    return cost, tour, elapsed

def print_results_table(res_pure, res_hybrid):
    """
    Calculates statistics and prints the comparison table.
    """
    cost_pure, _, time_pure = res_pure
    cost_hybrid, _, time_hybrid = res_hybrid
    
    print("\n" + "="*60)
    print(" FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} | {'Pure MMAS':<15} | {'Hybrid MMAS':<15}")
    print("-" * 60)
    print(f"{'Best Cost':<20} | {cost_pure:<15.2f} | {cost_hybrid:<15.2f}")
    print(f"{'Time (seconds)':<20} | {time_pure:<15.2f} | {time_hybrid:<15.2f}")
    
    cost_imp = cost_pure - cost_hybrid
    cost_imp_pct = (cost_imp / cost_pure) * 100 if cost_pure > 0 else 0
    
    if time_pure > 0:
        time_penalty = time_hybrid / time_pure
    else:
        time_penalty = 0

    print("-" * 60)
    print(f"Cost Improvement:    {cost_imp:.2f} ({cost_imp_pct:.2f}%)")
    print(f"Time Factor:         Hybrid is {time_penalty:.1f}x slower")
    print("="*60)

# =============================================================================
#  WRAPPER & MAIN
# =============================================================================

def run_full_comparison_for_file(filename, max_iters=200):
    """
    Executes the full comparison pipeline using distinct, scientifically tuned parameters.
    """
    # 1. Load Data
    try:
        if not os.path.exists(filename):
            print(f"[Error] File not found: {filename}")
            return
        instance = CVRPInstance(filename)
    except Exception as e:
        print(f"[Error] Could not load {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f" EXPERIMENT: {instance.name} (N={instance.n_locations})")
    print("="*60)

    # 2. Run Experiment A: Pure MMAS (Using Standard Parameters)
    results_pure = run_single_experiment(
        instance, 
        PARAMS_STANDARD,      # <--- Uses Low Rho, High Beta
        use_local_search=False, 
        max_iters=max_iters
    )

    # 3. Run Experiment B: Hybrid MMAS (Using Hybrid Parameters)
    results_hybrid = run_single_experiment(
        instance, 
        PARAMS_HYBRID,        # <--- Uses High Rho, Low Beta
        use_local_search=True, 
        max_iters=max_iters
    )

    # 4. Report & Visualize
    print_results_table(results_pure, results_hybrid)
    
    print(f"Generating Comparison Plot for {filename}...")
    plot_comparison(
        instance, 
        results_pure[1], results_pure[0], "Pure MMAS\n(Standard Physics)", 
        results_hybrid[1], results_hybrid[0], "Hybrid MMAS\n(Lamarckian Physics)"
    )

def main():
    # --- FINAL TEST SUITE ---
    
    # 1. Random Topology (Validation)
    run_full_comparison_for_file("./datasets/CMT1.vrp", max_iters=200)

    # 2. Clustered Topology (Stability Test)
    run_full_comparison_for_file("./datasets/CMT12.vrp", max_iters=200)
    

if __name__ == "__main__":
    main()