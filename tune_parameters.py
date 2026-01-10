import optuna
from cvrp_data import CVRPInstance
from solver import MMASSolver

def objective(trial):
    # 1. Define the search space
    alpha = trial.suggest_float("alpha", 0.5, 5.0)
    beta = trial.suggest_float("beta", 1.0, 8.0)
    rho = trial.suggest_float("rho", 0.01, 0.2) # MMAS likes low rho
    n_ants = trial.suggest_categorical("n_ants", [20, 30, 50])

    # 2. Setup the problem (Use a moderate sized instance like cmt01)
    # Note: Loading the file inside the loop is slow; load it globally if possible
    instance = CVRPInstance("E-n51-k5.vrp") 

    # 3. Run the Solver with these parameters
    # Reduce iterations for tuning to save time (e.g., 50 or 100)
    solver = MMASSolver(instance, n_ants=n_ants, rho=rho, alpha=alpha, beta=beta)
    cost, _ = solver.solve(max_iterations=500) 
    
    return cost

if __name__ == "__main__":
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50) # Run 50 different experiments

    print("Best params found:")
    print(study.best_params)
    
    # Optional: Visualize the search
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()