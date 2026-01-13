import os
import glob
import json
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
    
    # Merge CSVs
    df_list = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Load Convergence Data (Try to find CMT12 for consistency with previous studies)
    conv_data = {}
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*_convergence.json"))
    
    target_json = next((f for f in json_files if "CMT12" in f), None)
    if not target_json and json_files:
        target_json = json_files[0] # Fallback

    if target_json:
        with open(target_json, "r") as f:
            conv_data = json.load(f)
        
        base_name = os.path.basename(target_json)
        inst_name = base_name.replace('_convergence.json', '')
        conv_data['meta_instance_name'] = inst_name
            
    return full_df, conv_data

def perform_analysis(df, conv_data):
    print("\n" + "="*60)
    print(" SCIENTIFIC ANALYSIS REPORT (Demšar, 2006)")
    print("="*60)

    # --- 1. DATA PREPARATION & GAP CALCULATION ---
    # We calculate the 'Best Found' cost per instance to use as a fallback BKS
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

    # --- 3. WILCOXON SIGNED-RANKS TESTS ---
    # Demšar (2006) recommends Wilcoxon for comparing two classifiers over multiple datasets.
    # We aggregate to 1 mean value per instance per algorithm before testing.
    
    if len(instances) < 6:
        print("\n[Warning] Sample size (N < 6) is small for Wilcoxon test reliability.")

    # A. Solution Quality (Gap)
    agg_gap = df.groupby(['Instance', 'Algorithm'])['Gap_Pct'].mean().unstack()
    # Test: Is Standard Gap > Hybrid Gap? (meaning Hybrid is better/smaller gap)
    stat_q, p_q = wilcoxon(agg_gap['Standard'], agg_gap['Hybrid'], alternative='greater')
    
    print(f"\n[A] SOLUTION QUALITY (Gap Analysis)")
    print(f"    Wilcoxon p-value: {p_q:.6f}")
    if p_q < 0.05:
        print("    >> RESULT: Hybrid MMAS is SIGNIFICANTLY better (p < 0.05).")
    else:
        print("    >> RESULT: No significant difference detected.")

    # B. Computational Cost (Time)
    # Demšar: "They can be applied to... computation times." 
    agg_time = df.groupby(['Instance', 'Algorithm'])['Time'].mean().unstack()
    # Test: Is Hybrid Time > Standard Time?
    stat_t, p_t = wilcoxon(agg_time['Hybrid'], agg_time['Standard'], alternative='greater')
    
    print(f"\n[B] COMPUTATIONAL COST (Time Analysis)")
    print(f"    Wilcoxon p-value: {p_t:.6f}")
    if p_t < 0.05:
        print("    >> RESULT: Hybrid MMAS is SIGNIFICANTLY slower (p < 0.05).")
    else:
        print("    >> RESULT: Time difference is not significant.")

    # --- 4. VISUALIZATION ---
    print("\nGenerating Scientific Plots...")

    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    
    # Plot 1: Stability Box Plot (Quality)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Instance', y='Gap_Pct', hue='Algorithm', data=df, palette='Set2')
    plt.title('Solution Quality Distribution (Gap to BKS)')
    plt.ylabel('Gap % (Lower is Better)')
    plt.xlabel('Dataset Instance')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scientific_gap_boxplot.png'))
    print(f"  [Saved] {os.path.join(PLOTS_DIR, 'scientific_gap_boxplot.png')}")

    # Plot 2: Scalability Scatter Plot (Size vs Time)
    # This visualizes how time complexity grows with problem size
    plt.figure(figsize=(10, 4))
    sns.scatterplot(
        x='Size', y='Time', hue='Algorithm', style='Algorithm', 
        data=df, s=100, alpha=0.8
    )

    label_coords = df.groupby('Instance')[['Size', 'Time']].mean().reset_index()
    
    for i, row in label_coords.iterrows():
        plt.text(
            x=row['Size'], 
            y=row['Time'], 
            s=row['Instance'], 
            fontsize=9,
            color='black',
            ha='right',   # Align text to the right of the point (or 'center')
            va='bottom',  # Place text slightly above the point
            #alpha=0.9     # Slight transparency to not block data completely
        )

    # plt.yscale('log')  # Sets the Y-axis to logarithmic
    plt.title('Scalability: Execution Time vs Problem Size')
    plt.ylabel('Time (s)')
    # Use 'both' to show minor grid lines, which helps read log plots
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.xlabel('Number of Cities (N)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scientific_scalability.png'))
    print(f"  [Saved] {os.path.join(PLOTS_DIR, 'scientific_scalability.png')}")

    # Plot 3: Convergence Profile
    if conv_data:
        plt.figure(figsize=(10, 4))

        inst_title = conv_data.get('meta_instance_name', 'Representative Instance')
        # --- NEW: Retrieve REAL BKS from the dataset header ---
        # We look up the 'BKS' column in the DataFrame for this specific instance
        real_bks = -1
        subset = df[df['Instance'] == inst_title]
        if not subset.empty:
            real_bks = subset['BKS'].iloc[0]

        if 'Standard' in conv_data:
            plt.plot(conv_data['Standard'], label='Standard MMAS', color='blue', alpha=0.7)
        if 'Hybrid' in conv_data:
            plt.plot(conv_data['Hybrid'], label='Hybrid MMAS', color='red', linestyle='--')
        
        if real_bks > 0:
            plt.axhline(y=real_bks, color='green', linestyle=':', linewidth=2, label=f'Real BKS ({real_bks:.2f})')

        plt.title(f'Convergence Profile (Representative Instance: {inst_title})')
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost Found')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'scientific_convergence.png'))
        print(f"  [Saved] {os.path.join(PLOTS_DIR, 'scientific_convergence.png')}")

if __name__ == "__main__":
    df, conv = load_all_data()
    if df is not None:
        perform_analysis(df, conv)