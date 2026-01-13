import os
import time
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import wilcoxon

# Import your project modules
from cvrp_data import CVRPInstance
from solver import MMASSolver

# =============================================================================
#  CONFIGURATION (Scientifically Tuned)
# =============================================================================

# Parameters from run_single_experiment.py
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

# Execution Budgets
MAX_ITERS_STD = 200
MAX_ITERS_HYB = 25 

# Output Directory
RESULTS_DIR = "demo"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# =============================================================================
#  CORE SOLVER WRAPPER
# =============================================================================

def run_trial_unified(instance, params, use_ls, max_iters):
    """
    Unified runner that returns EVERYTHING needed for both modes:
    Cost, Time, Convergence History, and the Tour (Route).
    """
    n_ants = max(10, int(round(instance.n_locations * params['ant_factor'])))
    
    solver = MMASSolver(
        instance, n_ants=n_ants, rho=params['rho'], 
        alpha=params['alpha'], beta=params['beta'], use_local_search=use_ls
    )
    
    t0 = time.time()
    # Ensure solver.py returns: cost, tour, history
    cost, tour, history = solver.solve(max_iterations=max_iters, verbose=False)
    elapsed = time.time() - t0
    
    # Fix history format if needed (convert numpy types to float)
    if history and isinstance(history[0], (tuple, list)):
        # If tuple (iter, cost)
        history_clean = [(int(i), float(c)) for i, c in history]
    else:
        # If flat list, assume index is iteration
        history_clean = [(i, float(c)) for i, c in enumerate(history)]

    return float(cost), float(elapsed), history_clean, tour

# =============================================================================
#  MODE 1: SINGLE RUN VISUALIZATION
# =============================================================================

def plot_comparison(instance, tour1, cost1, title1, tour2, cost2, title2):
    """Plots two solutions side-by-side."""
    coords = instance.coords
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    titles = [f"{title1}\nCost: {cost1:.2f}", f"{title2}\nCost: {cost2:.2f}"]
    tours = [tour1, tour2]
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        
        # Plot Depot and Customers
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
            
            # Arrows
            ax.quiver(x_seq[:-1], y_seq[:-1], 
                      [x_seq[j+1]-x_seq[j] for j in range(len(x_seq)-1)],
                      [y_seq[j+1]-y_seq[j] for j in range(len(y_seq)-1)], 
                      scale_units='xy', angles='xy', scale=1, color='black', width=0.003, headwidth=4)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    # Optional: Save this plot to demo folder as well
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, f"{instance.name}_single_run_comparison.png")
    plt.savefig(save_path)
    print(f"  [Saved] Comparison plot to: {save_path}")
    
    plt.show()

def run_single_mode(instance, filename):
    print("\n" + "="*60)
    print(f" [SINGLE MODE] Visualizing 1 Run for: {filename}")
    print("="*60)

    # 1. Standard Run
    print("Running Standard MMAS...")
    c1, t1, _, tour1 = run_trial_unified(instance, PARAMS_STANDARD, False, MAX_ITERS_STD)

    # 2. Hybrid Run
    print("Running Hybrid MMAS...")
    c2, t2, _, tour2 = run_trial_unified(instance, PARAMS_HYBRID, True, MAX_ITERS_HYB)

    # 3. Print Stats
    print("\n" + "-"*50)
    print(f"{'Metric':<15} | {'Standard':<12} | {'Hybrid':<12}")
    print("-" * 50)
    print(f"{'Cost':<15} | {c1:<12.2f} | {c2:<12.2f}")
    print(f"{'Time (s)':<15} | {t1:<12.2f} | {t2:<12.2f}")
    
    if t1 > 0:
        print(f"Time Factor:     Hybrid is {t2/t1:.1f}x slower")
    if c1 > 0:
        imp = ((c1 - c2) / c1) * 100
        print(f"Improvement:     {imp:.2f}%")
    print("-" * 50)

    # 4. Plot
    print("Opening Comparison Plot...")
    plot_comparison(
        instance, 
        tour1, c1, "Standard MMAS", 
        tour2, c2, "Hybrid MMAS"
    )

# =============================================================================
#  MODE 2: SCIENTIFIC STATISTICAL MODE
# =============================================================================

def plot_convergence_aggregation(ax, history_list, label, color, max_iters):
    """Aggregates multiple sparse convergence histories into a Mean+StdDev plot."""
    if not history_list: return

    # 1. Densify (Sparse tuples -> Dense Array)
    dense_matrix = np.zeros((len(history_list), max_iters))
    
    for i, trial_data in enumerate(history_list):
        lookup = dict(trial_data) # Convert [(iter, cost)] to dict
        current_val = trial_data[0][1] if trial_data else 0
        
        for t in range(max_iters):
            if t in lookup:
                current_val = lookup[t]
            dense_matrix[i, t] = current_val

    # 2. Stats
    mean_curve = np.mean(dense_matrix, axis=0)
    std_curve = np.std(dense_matrix, axis=0)
    x_axis = np.arange(max_iters)

    # 3. Plot
    ax.plot(x_axis, mean_curve, color=color, linewidth=2, label=f"{label} (Mean)")
    ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)

def perform_analysis_plots(df, conv_data, filename):
    """Generates Boxplot, Scalability, and Convergence plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base_name = filename.replace('.vrp', '')

    # --- 1. Boxplot (Stability) ---
    plt.figure(figsize=(8, 6))
    # FIXED: Added hue='Algorithm' and legend=False to silence warning
    sns.boxplot(x='Algorithm', y='Cost', hue='Algorithm', legend=False, data=df, palette='Set2')
    plt.title(f'Cost Distribution (N={len(df)//2} runs): {base_name}')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_name}_boxplot.png"))
    print(f"  [Saved] {base_name}_boxplot.png")
    plt.close()

    # --- 2. Time Scatter (Scalability Placeholder) ---
    plt.figure(figsize=(8, 5))
    # FIXED: Added hue='Algorithm' and legend=False to silence warning
    sns.stripplot(x='Algorithm', y='Time', hue='Algorithm', legend=False, data=df, size=8, alpha=0.7, palette='Set1')
    plt.title(f'Computation Time Variance: {base_name}')
    plt.ylabel('Time (s)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_name}_timeplot.png"))
    print(f"  [Saved] {base_name}_timeplot.png")
    plt.close()

    # --- 3. Convergence (Mean + Std) ---
    plt.figure(figsize=(10, 5))
    
    # Plot Standard
    if 'Standard' in conv_data:
        plot_convergence_aggregation(plt.gca(), conv_data['Standard'], 'Standard', 'blue', MAX_ITERS_STD)
    
    # Plot Hybrid
    if 'Hybrid' in conv_data:
        plot_convergence_aggregation(plt.gca(), conv_data['Hybrid'], 'Hybrid', 'red', MAX_ITERS_HYB)
    
    # Plot Real BKS
    real_bks = df['BKS'].iloc[0]
    if real_bks > 0:
         plt.axhline(y=real_bks, color='green', linestyle=':', linewidth=2, label=f'BKS ({real_bks:.2f})')

    plt.title(f'Convergence Profile (Mean ± Std Dev): {base_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_name}_convergence.png"))
    print(f"  [Saved] {base_name}_convergence.png")
    plt.close()

def run_scientific_mode(instance, filename, n_trials):
    print("\n" + "="*60)
    print(f" [SCIENTIFIC MODE] Running {n_trials} Trials for: {filename}")
    print("="*60)
    
    detailed_data = []
    convergence_data = {'Standard': [], 'Hybrid': []}
    
    # Try to find BKS
    real_bks = instance.bks if (hasattr(instance, 'bks') and instance.bks) else -1.0

    # --- 1. Run Loop ---
    # Standard
    for i in tqdm(range(n_trials), desc="Standard MMAS", ncols=80):
        c, t, h, _ = run_trial_unified(instance, PARAMS_STANDARD, False, MAX_ITERS_STD)
        detailed_data.append({
            'Algorithm': 'Standard', 'Cost': c, 'Time': t, 'BKS': real_bks
        })
        convergence_data['Standard'].append(h)

    # Hybrid
    for i in tqdm(range(n_trials), desc="Hybrid MMAS  ", ncols=80):
        c, t, h, _ = run_trial_unified(instance, PARAMS_HYBRID, True, MAX_ITERS_HYB)
        detailed_data.append({
            'Algorithm': 'Hybrid', 'Cost': c, 'Time': t, 'BKS': real_bks
        })
        convergence_data['Hybrid'].append(h)

    # --- 2. Save Data ---
    df = pd.DataFrame(detailed_data)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base_name = filename.replace('.vrp', '')
    
    df.to_csv(os.path.join(RESULTS_DIR, f"{base_name}_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{base_name}_convergence.json"), "w") as f:
        json.dump(convergence_data, f)
        
    print(f"\n[Data Saved] to '{RESULTS_DIR}' folder")

    # --- 3. Statistical Test (Wilcoxon) ---
    print("\n[Statistical Analysis]")
    std_costs = df[df['Algorithm'] == 'Standard']['Cost']
    hyb_costs = df[df['Algorithm'] == 'Hybrid']['Cost']
    
    if len(std_costs) == len(hyb_costs):
        stat, p = wilcoxon(std_costs, hyb_costs, alternative='greater')
        print(f"  Wilcoxon Test (Standard > Hybrid?): p-value = {p:.6f}")
        if p < 0.05:
            print("  >> RESULT: Hybrid is SIGNIFICANTLY better.")
        else:
            print("  >> RESULT: No significant difference.")
    
    # --- 4. Generate Plots ---
    print("\n[Generating Plots]")
    perform_analysis_plots(df, convergence_data, filename)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ACO Experiment Runner (Unified)")
    
    # Required: Dataset
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the VRP file (e.g., CMT1.vrp) inside ./datasets/")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=['single', 'scientific'], default='single',
                        help="'single' = 1 run with Route Plot. 'scientific' = N trials with Stats Plots.")
    
    # Optional: Trials (only for scientific mode)
    parser.add_argument("--trials", type=int, default=20, 
                        help="Number of trials for scientific mode (default: 20)")

    args = parser.parse_args()

    # Load Instance
    file_path = os.path.join("./datasets", args.dataset)
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return
    
    try:
        instance = CVRPInstance(file_path)
    except Exception as e:
        print(f"[Error] Failed to load instance: {e}")
        return

    # Dispatch
    if args.mode == 'single':
        run_single_mode(instance, args.dataset)
    else:
        run_scientific_mode(instance, args.dataset, args.trials)

if __name__ == "__main__":
    main()