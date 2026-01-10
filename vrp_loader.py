from cvrp_data import CVRPInstance
import matplotlib.pyplot as plt

def plot_instance(cvrp_instance):
    coords = cvrp_instance.coords
    demands = cvrp_instance.demands

    # Separate Depot (Index 0) from Customers (Index 1+)
    depot_x, depot_y = coords[0]
    cust_x = [c[0] for c in coords[1:]]
    cust_y = [c[1] for c in coords[1:]]

    plt.figure(figsize=(10, 8))
    
    # Plot Depot (Red Square)
    plt.scatter(depot_x, depot_y, c='red', marker='s', s=100, label='Depot')
    
    # Plot Customers (Blue Dots)
    plt.scatter(cust_x, cust_y, c='blue', s=30, alpha=0.6, label='Customers')

    # Optional: Annotate Demand on nodes
    for i in range(1, len(coords)):
        plt.text(coords[i][0], coords[i][1], str(demands[i]), fontsize=9)

    plt.title(f"CVRP Instance: {cvrp_instance.name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# 1. Define the filename
file_path = "E-n51-k5.vrp" 

# 2. Initialize the instance
try:
    cvrp = CVRPInstance(file_path)
    
    # 3. Verify data loaded correctly
    print(f"Instance Name: {cvrp.name}")
    print(f"Total Locations (N): {cvrp.n_locations}")
    print(f"Vehicle Capacity: {cvrp.capacity}")
    
    # Check the first few coordinates (usually the Depot is first)
    print(f"Depot Coordinates: {cvrp.coords[0]}")
    print(f"Depot Demand: {cvrp.demands[0]}") # Should be 0
    
    # Get data for the solver
    dist_matrix, demands, capacity = cvrp.get_data()
    print("\nSuccess! Data is ready for the solver.")
    print(f"Distance Matrix Shape: {dist_matrix.shape}")

    plot_instance(cvrp)

except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Make sure you downloaded the file.")
except Exception as e:
    print(f"An error occurred: {e}")