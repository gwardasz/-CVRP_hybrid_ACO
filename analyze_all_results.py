import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

RESULTS_DIR = "results"

def load_all_data():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*_results.csv"))
    
    if not csv_files:
        print(f"[Error] No result files found in '{RESULTS_DIR}'. Run experiments first.")
        return None, None

    print(f"Found {len(csv_files)} dataset result files. Merging...")
    
    df_list = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Load ALL Convergence Data files into a dictionary
    all_conv_data = {}
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*_convergence.json"))
    
    print(f"Found {len(json_files)} convergence history files.")
    
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            base_name = os.path.basename(jf)
            inst_name = base_name.replace('_convergence.json', '')
            
            all_conv_data[inst_name] = data
        except Exception as e:
            print(f"[Warning] Failed to load {jf}: {e}")
            
    return full_df, all_conv_data

def plot_convergence_lines(ax, history_list, label, color, max_iters):
    """
    Plots individual trial lines (Messy but shows raw behavior).
    """
    # Plot each trial with low opacity
    for i, trial_data in enumerate(history_list):
        if not trial_data: continue
        
        iterations, costs = zip(*trial_data)
        iters = list(iterations)
        vals = list(costs)
        
        # Extend to max_iters for visualization
        if iters[-1] < max_iters:
            iters.append(max_iters)
            vals.append(vals[-1])
            
        lbl = label if i == 0 else "_nolegend_"
        ax.step(iters, vals, where='post', color=color, alpha=0.15, linewidth=1.5, label=lbl)

def plot_convergence_aggregation(ax, history_list, label, color, max_iters):
    """
    Plots the MEAN convergence curve with a Standard Deviation shadow.
    """
    if not history_list: return

    # 1. DENSIFY: Convert sparse tuples to dense arrays
    # Matrix shape: (N_Trials, Max_Iters)
    dense_matrix = np.zeros((len(history_list), max_iters))
    
    for i, trial_data in enumerate(history_list):
        # Convert list of tuples to a dict for easy lookup
        # trial_data = [(0, 100), (5, 90)...]
        lookup = dict(trial_data)
        
        # If iteration 0 is missing (unlikely), fallback to the first known value
        current_val = trial_data[0][1] if trial_data else 0
        
        # Forward Fill loop
        for t in range(max_iters):
            if t in lookup:
                current_val = lookup[t]
            dense_matrix[i, t] = current_val

    # 2. AGGREGATE: Calculate Mean and Std Dev
    mean_curve = np.mean(dense_matrix, axis=0)
    std_curve = np.std(dense_matrix, axis=0)
    x_axis = np.arange(max_iters)

    # 3. PLOT
    # Solid Mean Line
    ax.plot(x_axis, mean_curve, color=color, linewidth=2, label=f"{label} (Mean)")
    
    # Shaded Std Dev Area
    ax.fill_between(
        x_axis, 
        mean_curve - std_curve, 
        mean_curve + std_curve, 
        color=color, alpha=0.2, label='_nolegend_'
    )

def perform_analysis(df, conv_data, plot_mode):
    print("\n" + "="*60)
    print(" SCIENTIFIC ANALYSIS REPORT")
    print("="*60)

    # --- 1. DATA PREPARATION ---
    best_found_map = df.groupby('Instance')['Cost'].min().to_dict()
    
    def calc_gap(row):
        ref = row['BKS']
        if ref <= 0: ref = best_found_map[row['Instance']]
        return ((row['Cost'] - ref) / ref) * 100

    df['Gap_Pct'] = df.apply(calc_gap, axis=1)

    # --- 2. SUMMARY TABLE ---
    print(f"{'Instance':<15} | {'Std Cost':<10} | {'Hyb Cost':<10} | {'Time Factor':<12}")
    print("-" * 60)
    
    instances = df['Instance'].unique()
    for inst in instances:
        sub = df[df['Instance'] == inst]
        s_avg = sub[sub['Algorithm'] == 'Standard']['Cost'].mean()
        h_avg = sub[sub['Algorithm'] == 'Hybrid']['Cost'].mean()
        s_time = sub[sub['Algorithm'] == 'Standard']['Time'].mean()
        h_time = sub[sub['Algorithm'] == 'Hybrid']['Time'].mean()
        
        factor = h_time / s_time if s_time > 0 else 1.0
        print(f"{inst:<15} | {s_avg:<10.2f} | {h_avg:<10.2f} | {factor:.1f}x")

    # --- 3. STATISTICAL TESTS ---
    if len(instances) >= 6:
        agg_gap = df.groupby(['Instance', 'Algorithm'])['Gap_Pct'].mean().unstack()
        stat_q, p_q = wilcoxon(agg_gap['Standard'], agg_gap['Hybrid'], alternative='greater')
        
        print(f"\n[A] SOLUTION QUALITY (Gap Analysis)")
        print(f"    Wilcoxon p-value: {p_q:.6f}")
        if p_q < 0.05: print("    >> Hybrid is SIGNIFICANTLY better.")
        else: print("    >> No significant difference.")

        agg_time = df.groupby(['Instance', 'Algorithm'])['Time'].mean().unstack()
        stat_t, p_t = wilcoxon(agg_time['Hybrid'], agg_time['Standard'], alternative='greater')
        
        print(f"\n[B] COMPUTATIONAL COST")
        print(f"    Wilcoxon p-value: {p_t:.6f}")
        if p_t < 0.05: print("    >> Hybrid is SIGNIFICANTLY slower.")
    else:
        print("\n[Warning] N < 6, skipping Wilcoxon tests.")

    # --- 4. VISUALIZATION ---
    print("\nGenerating Scientific Plots...")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot 1: Boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Instance', y='Gap_Pct', hue='Algorithm', data=df, palette='Set2')
    plt.title('Solution Quality Distribution (Gap to BKS)')
    plt.ylabel('Gap %')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scientific_gap_boxplot.png'))

    # Plot 2: Scalability
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x='Size', y='Time', hue='Algorithm', style='Algorithm', data=df, s=100, alpha=0.8)
    
    label_coords = df.groupby('Instance')[['Size', 'Time']].mean().reset_index()
    for i, row in label_coords.iterrows():
        plt.text(x=row['Size'], y=row['Time'], s=row['Instance'], fontsize=9, ha='right', va='bottom')

    plt.title('Scalability: Execution Time vs Problem Size')
    plt.ylabel('Time (s)')
    plt.xlabel('Number of Cities (N)')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scientific_scalability.png'))

    # Plot 3: Convergence
    if conv_data:
        print(f"  [Plotting] Convergence plots ({plot_mode} mode) for {len(conv_data)} instances...")
        
        for inst_name, data in conv_data.items():
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Retrieve Limits
            meta = data.get('Meta_Max_Iters', {'Standard': 200, 'Hybrid': 25})
            
            # Retrieve Real BKS
            real_bks = -1
            subset = df[df['Instance'] == inst_name]
            if not subset.empty: real_bks = subset['BKS'].iloc[0]

            # PLOT LOGIC SELECTOR
            if plot_mode == 'mean':
                # Use the Aggregation function
                if 'Standard' in data:
                    plot_convergence_aggregation(ax, data['Standard'], 'Standard MMAS', 'blue', meta['Standard'])
                if 'Hybrid' in data:
                    plot_convergence_aggregation(ax, data['Hybrid'], 'Hybrid MMAS', 'red', meta['Hybrid'])
            
            elif plot_mode == 'all':
                # Plot all raw lines (Step function)
                if 'Standard' in data:
                    plot_convergence_lines(ax, data['Standard'], 'Standard MMAS', 'blue', meta['Standard'])
                if 'Hybrid' in data:
                    plot_convergence_lines(ax, data['Hybrid'], 'Hybrid MMAS', 'red', meta['Hybrid'])

            elif plot_mode == 'one':
                 # Plot single line
                if 'Standard' in data:
                     # Create a single-item list
                    single_list = [data['Standard'][0]] if data['Standard'] else []
                    plot_convergence_lines(ax, single_list, 'Standard MMAS', 'blue', meta['Standard'])
                if 'Hybrid' in data:
                    single_list = [data['Hybrid'][0]] if data['Hybrid'] else []
                    plot_convergence_lines(ax, single_list, 'Hybrid MMAS', 'red', meta['Hybrid'])

            if real_bks > 0:
                ax.axhline(y=real_bks, color='green', linestyle=':', linewidth=2, label=f'Real BKS ({real_bks:.2f})')

            ax.set_title(f'Convergence Profile: {inst_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Cost Found')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_name = f'scientific_convergence_{inst_name}_{plot_mode}.png'
            plt.savefig(os.path.join(PLOTS_DIR, save_name))
            plt.close()
            
        print(f"  [Saved] Convergence plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated choices to include 'mean'
    parser.add_argument("--plot-mode", choices=['all', 'one', 'mean'], default='mean', 
                        help="Plot 'mean' (average+std), 'all' (raw lines), or 'one' (first trial).")
    args = parser.parse_args()

    df, conv_dict = load_all_data()
    if df is not None:
        perform_analysis(df, conv_dict, args.plot_mode)