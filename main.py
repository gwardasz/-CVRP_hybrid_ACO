import matplotlib.pyplot as plt
import numpy as np
from cvrp_data import CVRPInstance
import time
# Import your solvers
from as_solver import ASSolver
from solver import MMASSolver

def plot_solution(cvrp_instance, tour, cost, title="Solution"):
    """
    Visualizes the final VRP route using Matplotlib.
    """
    coords = cvrp_instance.coords
    
    # 1. Setup Plot
    plt.figure(figsize=(10, 8))
    plt.title(f"{title} | Cost: {cost:.2f}")
    
    # 2. Plot Nodes
    # Depot (Index 0) - Red Square
    plt.scatter(coords[0][0], coords[0][1], c='red', marker='s', s=100, label='Depot', zorder=5)
    
    # Customers (Indices 1+) - Blue Dots
    cust_x = [c[0] for c in coords[1:]]
    cust_y = [c[1] for c in coords[1:]]
    plt.scatter(cust_x, cust_y, c='blue', s=30, alpha=0.6, label='Customers')

    # 3. Plot Routes
    # FIX: Check if tour is not None and has length > 0
    if tour is not None and len(tour) > 0:
        x_seq = [coords[node][0] for node in tour]
        y_seq = [coords[node][1] for node in tour]
        
        # Plot lines
        plt.plot(x_seq, y_seq, c='black', linewidth=1, alpha=0.7, linestyle='-')
        
        # Plot arrows
        plt.quiver(x_seq[:-1], y_seq[:-1], 
                   [x_seq[i+1]-x_seq[i] for i in range(len(x_seq)-1)],
                   [y_seq[i+1]-y_seq[i] for i in range(len(y_seq)-1)], 
                   scale_units='xy', angles='xy', scale=1, color='black', width=0.003)

    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(instance, tour1, cost1, title1, tour2, cost2, title2):
    """
    Plots two solutions side-by-side for easy visual comparison.
    """
    coords = instance.coords
    
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Data for plotting
    titles = [f"{title1}\nCost: {cost1:.2f}", f"{title2}\nCost: {cost2:.2f}"]
    tours = [tour1, tour2]
    
    # Loop to draw both plots
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        
        # 1. Plot Nodes
        # Depot (Red Square)
        ax.scatter(coords[0][0], coords[0][1], c='red', marker='s', s=100, label='Depot', zorder=5)
        # Customers (Blue Dots)
        cust_x = [c[0] for c in coords[1:]]
        cust_y = [c[1] for c in coords[1:]]
        ax.scatter(cust_x, cust_y, c='blue', s=30, alpha=0.6, label='Customers')
        
        # 2. Plot Routes
        tour = tours[i]
        if tour is not None and len(tour) > 0:
            # Connect the dots
            x_seq = [coords[node][0] for node in tour]
            y_seq = [coords[node][1] for node in tour]
            
            # Draw lines
            ax.plot(x_seq, y_seq, c='black', linewidth=1, alpha=0.7, linestyle='-')
            
            # Draw arrows (Optional: adds directional context)
            ax.quiver(x_seq[:-1], y_seq[:-1], 
                    [x_seq[j+1]-x_seq[j] for j in range(len(x_seq)-1)],
                    [y_seq[j+1]-y_seq[j] for j in range(len(y_seq)-1)], 
                    scale_units='xy', angles='xy', scale=1, color='black', width=0.003, headwidth=4)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



def main():
    # --- 1. Load Data ---
    filename = "E-n51-k5.vrp"
    print(f"Loading instance: {filename}")
    try:
        instance = CVRPInstance(filename)
        print(f"Loaded {instance.name} with {instance.n_locations} nodes.\n")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # --- 2. Configuration (From your Optuna Tuning) ---
    N_ANTS = 50
    RHO = 0.20
    ALPHA = 1.0
    BETA = 2.84
    MAX_ITERS = 5 #200 # Higher budget for final showcase
    
    # --- NEW: Print Experiment Configuration ---
    print("\n" + "="*60)
    print(" EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f" Instance:       {filename} ({instance.n_locations} nodes)")
    print(f" Max Iterations: {MAX_ITERS}")
    print("-" * 60)
    print(" MMAS PARAMETERS (Optimized):")
    print(f"   n_ants:       {N_ANTS}")
    print(f"   rho:          {RHO}  (Evaporation)")
    print(f"   alpha:        {ALPHA}   (Pheromone)")
    print(f"   beta:         {BETA}  (Heuristic)")
    print("-" * 60)
    print(" LOCAL SEARCH (VND) SEQUENCE:")
    print("   1. Relocate   (Inter-Route / Load Balancing)")
    print("   2. Swap       (Inter-Route / Clustering)")
    print("   3. 2-Opt      (Intra-Route / Uncrossing)")
    print("   4. 2-Opt* (Inter-Route / Tail Swap)")
    print("="*60 + "\n")

    # --- 3. Run Pure MMAS (Baseline) ---
    print(">>> Running Pure MMAS (No Local Search)...")
    t0 = time.time()
    solver_pure = MMASSolver(
        instance, n_ants=N_ANTS, rho=RHO, alpha=ALPHA, beta=BETA, 
        use_local_search=False # <--- DISABLED
    )
    # verbose=True shows the progress bar we added
    cost_pure, tour_pure = solver_pure.solve(max_iterations=MAX_ITERS, verbose=True)
    time_pure = time.time() - t0

    # --- 4. Run Hybrid MMAS (VND) ---
    print("\n>>> Running Hybrid MMAS (With VND Local Search)...")
    t0 = time.time()
    solver_hybrid = MMASSolver(
        instance, n_ants=N_ANTS, rho=RHO, alpha=ALPHA, beta=BETA, 
        use_local_search=True  # <--- ENABLED
    )
    cost_hybrid, tour_hybrid = solver_hybrid.solve(max_iterations=MAX_ITERS, verbose=True)
    time_hybrid = time.time() - t0

    # --- 5. Report Statistics ---
    print("\n" + "="*60)
    print(" FINAL COMPARISON RESULTS")
    print("="*60)
    
    # Header
    print(f"{'Metric':<20} | {'Pure MMAS':<15} | {'Hybrid MMAS':<15}")
    print("-" * 60)
    
    # Cost Comparison
    print(f"{'Best Cost':<20} | {cost_pure:<15.2f} | {cost_hybrid:<15.2f}")
    
    # Time Comparison (NEW)
    print(f"{'Time (seconds)':<20} | {time_pure:<15.2f} | {time_hybrid:<15.2f}")
    
    # Calculate Improvements
    cost_imp = cost_pure - cost_hybrid
    cost_imp_pct = (cost_imp / cost_pure) * 100
    
    # Calculate Time Penalty (NEW)
    if time_pure > 0:
        time_penalty_factor = time_hybrid / time_pure
    else:
        time_penalty_factor = 0

    print("-" * 60)
    print(f"Cost Improvement:    {cost_imp:.2f} ({cost_imp_pct:.2f}%)")
    print(f"Time Factor:         Hybrid is {time_penalty_factor:.1f}x slower")
    print("="*60)


    # --- 6. Visualize Differences ---
    print("Generating Comparison Plot...")
    plot_comparison(
        instance, 
        tour_pure, cost_pure, "Pure MMAS", 
        tour_hybrid, cost_hybrid, "Hybrid MMAS (VND)"
    )
    # plot_solution(instance, tour_hybrid, cost_hybrid, title="Hybrid MMAS Solution")


if __name__ == "__main__":
    main()