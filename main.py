import matplotlib.pyplot as plt
from cvrp_data import CVRPInstance

# Import your solvers
from as_solver import ASSolver
from solver import MMASSolver

# Uncomment this line once you save the MMAS code into MMAS_solver.py
# from MMAS_solver import MMASSolver 

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
    # The 'tour' is a list of indices, e.g., [0, 5, 10, 0, 12, 0]
    # We draw lines connecting the sequence
    if tour:
        x_seq = [coords[node][0] for node in tour]
        y_seq = [coords[node][1] for node in tour]
        
        # Plot arrows/lines
        plt.plot(x_seq, y_seq, c='black', linewidth=1, alpha=0.7, linestyle='-')
        
        # Optional: Add arrows to show direction
        plt.quiver(x_seq[:-1], y_seq[:-1], 
                   [x_seq[i+1]-x_seq[i] for i in range(len(x_seq)-1)],
                   [y_seq[i+1]-y_seq[i] for i in range(len(y_seq)-1)], 
                   scale_units='xy', angles='xy', scale=1, color='black', width=0.003)

    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # --- 1. Load Data ---
    filename = "E-n51-k5.vrp"
    print(f"Loading instance: {filename}")
    try:
        instance = CVRPInstance(filename)
        print(f"Loaded {instance.name} with {instance.n_locations} nodes.\n")
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please download it to this folder.")
        return

    # --- 2. Select Solver (Phase 1 vs Phase 2) ---
    
    # === PHASE 1: Ant System (AS) ===
    # High rho (0.5) because all ants deposit pheromones

    #solver = ASSolver(instance, n_ants=300, rho=0.5, alpha=1.0, beta=2.5)
    #algorithm_name = "Ant System (Baseline)"

    # === PHASE 2: MMAS (Uncomment below to run MMAS) ===
    solver = MMASSolver(instance, n_ants=50, rho=0.16, alpha=0.97, beta=4.82)
    algorithm_name = "MMAS (Enhanced)"

    # --- 3. Run Optimization ---
    print(f"Running {algorithm_name}...")
    
    # You can increase max_iterations for better results (e.g., 200 or 500)
    best_cost, best_tour = solver.solve(max_iterations=500)

    # --- 4. Report Results ---
    print("-" * 50)
    print(f"Optimization Complete.")
    print(f"Algorithm: {algorithm_name}")
    print(f"Final Best Cost: {best_cost:.2f}")
    print(f"Best Tour (Sequence): {best_tour}")
    print("-" * 50)

    # --- 5. Visualize ---
    plot_solution(instance, best_tour, best_cost, title=algorithm_name)

if __name__ == "__main__":
    main()