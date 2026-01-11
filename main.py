import matplotlib.pyplot as plt
import numpy as np
import time
from cvrp_data import CVRPInstance
from solver import MMASSolver

# --- VISUALIZATION FUNCTIONS ---

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
            ax.plot(x_seq, y_seq, c='black', linewidth=1, alpha=0.7, linestyle='-')
            ax.quiver(x_seq[:-1], y_seq[:-1], 
                      [x_seq[j+1]-x_seq[j] for j in range(len(x_seq)-1)],
                      [y_seq[j+1]-y_seq[j] for j in range(len(y_seq)-1)], 
                      scale_units='xy', angles='xy', scale=1, color='black', width=0.003, headwidth=4)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# --- CORE LOGIC FUNCTIONS ---

def run_single_experiment(instance, params, use_local_search, max_iters, verbose=True):
    """
    Runs a single MMAS experiment and returns the results.
    Returns: (cost, tour, elapsed_time)
    """
    if verbose:
        mode = "Hybrid (VND)" if use_local_search else "Pure MMAS"
        print(f"\n>>> Running {mode}...")
    
    t0 = time.time()
    
    solver = MMASSolver(
        instance, 
        n_ants=params['n_ants'], 
        rho=params['rho'], 
        alpha=params['alpha'], 
        beta=params['beta'], 
        use_local_search=use_local_search
    )
    
    cost, tour = solver.solve(max_iterations=max_iters, verbose=verbose)
    elapsed = time.time() - t0
    
    return cost, tour, elapsed

def print_experiment_config(filename, n_nodes, max_iters, params):
    """Prints the professional configuration header."""
    print("\n" + "="*60)
    print(f" EXPERIMENT CONFIGURATION: {filename}")
    print("="*60)
    print(f" Instance:       {filename} ({n_nodes} nodes)")
    print(f" Max Iterations: {max_iters}")
    print("-" * 60)
    print(" MMAS PARAMETERS:")
    print(f"   n_ants:       {params['n_ants']}")
    print(f"   rho:          {params['rho']}  (Evaporation)")
    print(f"   alpha:        {params['alpha']}   (Pheromone)")
    print(f"   beta:         {params['beta']}  (Heuristic)")
    print("-" * 60)
    print(" LOCAL SEARCH (VND):")
    print("   1. Relocate -> 2. Swap -> 3. 2-Opt -> 4. 2-Opt*")
    print("="*60 + "\n")

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
    cost_imp_pct = (cost_imp / cost_pure) * 100
    
    if time_pure > 0:
        time_penalty = time_hybrid / time_pure
    else:
        time_penalty = 0

    print("-" * 60)
    print(f"Cost Improvement:    {cost_imp:.2f} ({cost_imp_pct:.2f}%)")
    print(f"Time Factor:         Hybrid is {time_penalty:.1f}x slower")
    print("="*60)

# --- WRAPPER FUNCTION ---

def run_full_comparison_for_file(filename, max_iters=200):
    """
    Wrapper function to execute the full comparison pipeline for a given dataset file.
    """
    # 1. Load Data
    try:
        instance = CVRPInstance(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # 2. Setup Parameters (Defaults from tuning)
    PARAMS = {
        'n_ants': 50,
        'rho': 0.20,
        'alpha': 1.0,
        'beta': 2.84
    }

    # 3. Print Config
    print_experiment_config(filename, instance.n_locations, max_iters, PARAMS)

    # 4. Run Experiment A: Pure MMAS
    results_pure = run_single_experiment(
        instance, PARAMS, 
        use_local_search=False, 
        max_iters=max_iters
    )

    # 5. Run Experiment B: Hybrid MMAS
    results_hybrid = run_single_experiment(
        instance, PARAMS, 
        use_local_search=True, 
        max_iters=max_iters
    )

    # 6. Report & Visualize
    print_results_table(results_pure, results_hybrid)
    
    print(f"Generating Comparison Plot for {filename}...")
    plot_comparison(
        instance, 
        results_pure[1], results_pure[0], "Pure MMAS", 
        results_hybrid[1], results_hybrid[0], "Hybrid MMAS (VND)"
    )

# --- MAIN EXECUTION ---

def main():
    # You can now easily run multiple files here!
    
    # Run 1: The Standard Benchmark
    run_full_comparison_for_file("CMT4.vrp", max_iters=200)

    # Example: Run 2 (Uncomment if you have another file)
    # run_full_comparison_for_file("A-n32-k5.vrp", max_iters=200)

if __name__ == "__main__":
    main()