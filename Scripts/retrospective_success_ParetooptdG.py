import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_and_plot_optimized_cutoffs(file_path, min_designs=5, user_dG_cutoff=3.37): #Input desired custom dG cutoff here
    # Load dataset
    df = pd.read_csv(file_path)

    # Clean and parse numerical columns
    df_valid = df.dropna(subset=['Binding', 'dG_ensemble', 'Length']).copy()
    df_valid['Binding'] = df_valid['Binding'].astype(int)
    df_valid['dG_ensemble'] = df_valid['dG_ensemble'].astype(float)
    df_valid['Length'] = df_valid['Length'].astype(float)

    # Bin designs by Length
    def get_length_bin(length):
        if length < 80:
            return '< 80'
        elif 80 <= length <= 100:
            return '80 - 100'
        else:
            return '> 100'

    df_valid['Length_Bin'] = df_valid['Length'].apply(get_length_bin)

    bins = ['All Data', '< 80', '80 - 100', '> 100']
    optimization_results = []
    pareto_data_per_bin = {}

    # Candidate cutoffs to evaluate across metric range
    min_val = df_valid['dG_ensemble'].min()
    max_val = df_valid['dG_ensemble'].max()
    candidate_cutoffs = np.linspace(min_val, max_val, 100)

    for b in bins:
        subset = df_valid if b == 'All Data' else df_valid[df_valid['Length_Bin'] == b]
        
        # Baseline total designs and total true binders
        total_baseline = len(subset)
        success_baseline = int(subset['Binding'].sum())
        baseline_rate = (success_baseline / total_baseline * 100) if total_baseline > 0 else 0.0

        # Collect candidate cutoffs that improve upon baseline success rate
        candidates = []
        for cutoff in candidate_cutoffs:
            passing = subset[subset['dG_ensemble'] > cutoff]
            n_passing = len(passing)
            
            if n_passing >= min_designs:
                n_succ = passing['Binding'].sum()
                rate = (n_succ / n_passing) * 100
                
                if rate >= baseline_rate:
                    candidates.append({
                        'cutoff': cutoff,
                        'rate': rate,
                        'n_succ': int(n_succ),
                        'n_passing': n_passing
                    })

        # Identify Pareto optimal solution
        if candidates:
            cand_df = pd.DataFrame(candidates)
            
            rates = cand_df['rate'].values
            succs = cand_df['n_succ'].values
            is_pareto = []
            
            for i in range(len(cand_df)):
                dominated = False
                for j in range(len(cand_df)):
                    if i != j:
                        if (rates[j] >= rates[i] and succs[j] >= succs[i]) and (rates[j] > rates[i] or succs[j] > succs[i]):
                            dominated = True
                            break
                is_pareto.append(not dominated)
            
            cand_df['is_pareto'] = is_pareto
            pareto_df = cand_df[cand_df['is_pareto']].copy()

            min_r, max_r = pareto_df['rate'].min(), pareto_df['rate'].max()
            min_s, max_s = pareto_df['n_succ'].min(), pareto_df['n_succ'].max()
            
            r_range = (max_r - min_r) if max_r > min_r else 1.0
            s_range = (max_s - min_s) if max_s > min_s else 1.0
            
            norm_rate = (pareto_df['rate'] - min_r) / r_range
            norm_succ = (pareto_df['n_succ'] - min_s) / s_range
            
            dist_to_ideal = np.sqrt((1 - norm_rate)**2 + (1 - norm_succ)**2)
            best_idx = dist_to_ideal.idxmin()
            best_row = pareto_df.loc[best_idx]
            
            best_rate = best_row['rate']
            best_cutoff = best_row['cutoff']
            best_passing_count = int(best_row['n_passing'])
            best_success_count = int(best_row['n_succ'])
            
            pareto_data_per_bin[b] = (cand_df, best_row)
        else:
            best_rate = baseline_rate
            best_cutoff = min_val
            best_passing_count = total_baseline
            best_success_count = success_baseline
            pareto_data_per_bin[b] = (None, None)

        optimization_results.append({
            'Length Bin': b,
            'Baseline Total Count': total_baseline,
            'Baseline Success Count': success_baseline,
            'Baseline Success Rate (%)': round(baseline_rate, 2),
            'Optimal Cutoff (>)': round(best_cutoff, 2),
            'Optimized Success Rate (%)': round(best_rate, 2),
            'Passing Count': best_passing_count,
            'Success Count': best_success_count
        })

    results_df = pd.DataFrame(optimization_results)
    results_df.to_csv('Pareto_optimized_cutoffs_summary.csv', index=False)

    sns.set_theme(style="whitegrid")

    # =========================================================================
    # SECTION 1: INDIVIDUAL PARETO FRONTIER PLOTS PER BIN
    # =========================================================================
    for b in bins:
        cand_df, best_row = pareto_data_per_bin[b]
        if cand_df is None or cand_df.empty:
            continue

        pareto_df = cand_df[cand_df['is_pareto']].sort_values(by='n_succ').reset_index(drop=True)

        plt.figure(figsize=(8, 6))

        plt.scatter(
            cand_df[~cand_df['is_pareto']]['n_succ'], 
            cand_df[~cand_df['is_pareto']]['rate'], 
            color='gray', alpha=0.4, label='Sub-optimal Cutoffs'
        )

        plt.plot(
            pareto_df['n_succ'], pareto_df['rate'], 
            color='#1f77b4', linestyle='--', linewidth=2, label='Pareto Frontier'
        )
        plt.scatter(pareto_df['n_succ'], pareto_df['rate'], color='#1f77b4', s=50)

        plt.scatter(
            best_row['n_succ'], best_row['rate'], 
            color='#d62728', s=130, zorder=5, 
            label=f"Selected Optimal (Cutoff > {best_row['cutoff']:.2f} kcal/mol)"
        )

        sanitized_bin_name = b.replace(' ', '_').replace('<', 'less_than').replace('>', 'greater_than')
        filename = f"Anthropic_pareto_frontier_{sanitized_bin_name}.png"

        plt.xlabel("Number of Binders Recovered", fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.legend(loc='lower left', frameon=True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    # =========================================================================
    # SECTION 2: BAR PLOT WITH BIN-SPECIFIC OPTIMIZED CUTOFFS
    # =========================================================================
    plot_data = []
    for _, row in results_df.iterrows():
        plot_data.append({
            'Length Bin': row['Length Bin'],
            'Type': 'Overall Rate (Uncut)',
            'Success Rate (%)': row['Baseline Success Rate (%)'],
            'Cutoff': None,
            'Success Count': row['Baseline Success Count'],
            'Total Binders': row['Baseline Success Count']
        })
        plot_data.append({
            'Length Bin': row['Length Bin'],
            'Type': 'Optimized Cutoff',
            'Success Rate (%)': row['Optimized Success Rate (%)'],
            'Cutoff': row['Optimal Cutoff (>)'],
            'Success Count': row['Success Count'],
            'Total Binders': row['Baseline Success Count']
        })
    
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(11, 6))

    ax = sns.barplot(
        data=plot_df,
        x='Length Bin',
        y='Success Rate (%)',
        hue='Type',
        palette=['#4c72b0', '#55a868']
    )

    plt.title('Overall vs. Pareto Optimal dG_ensemble Success Rate', fontsize=13, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=11)
    plt.xlabel('Length Bin', fontsize=11)
    
    max_rate = plot_df['Success Rate (%)'].max()
    plt.ylim(0, max_rate + 25)
    plt.legend(title='Metric Threshold', frameon=True, loc='upper right')

    num_bins = len(bins)
    y_row_2 = max_rate + 12
    ax.text(-0.4, y_row_2, "Binders Retained:", ha='left', va='center', fontsize=9.5, fontweight='bold', color='#333333')

    for i, row in plot_df.iterrows():
        bin_idx = bins.index(row['Length Bin'])
        hue_idx = 0 if row['Type'] == 'Overall Rate (Uncut)' else 1
        
        patch_idx = hue_idx * num_bins + bin_idx
        p = ax.patches[patch_idx]
        
        height = p.get_height()
        if not np.isnan(height) and height >= 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', rotation=0, fontsize=9, xytext=(0, 3), textcoords='offset points'
            )
            
            if row['Type'] == 'Optimized Cutoff':
                succ = row['Success Count']
                total_binders = row['Total Binders']
                bar_center_x = p.get_x() + p.get_width() / 2.
                
                # Display fraction evenly across top row
                ax.text(
                    bar_center_x, y_row_2,
                    f"{succ}/{total_binders}",
                    ha='center', va='center', fontsize=9, fontweight='bold', color='#2e6930'
                )
                
                # Cutoff label displayed directly above bar percentage
                ax.annotate(
                    f"Cutoff > {row['Cutoff']:.2f}",
                    (bar_center_x, height + 3.0),
                    ha='center', va='bottom', rotation=0, fontsize=8, fontweight='bold', color='#2e6930'
                )

    plt.tight_layout()
    plt.savefig('Pareto_optimized_success_rate_comparison.png', dpi=300)
    plt.close()

    # =========================================================================
    # SECTION 3: RETROSPECTIVE BAR PLOT USING GLOBAL ('ALL DATA') CUTOFF
    # =========================================================================
    all_data_cutoff = results_df.loc[results_df['Length Bin'] == 'All Data', 'Optimal Cutoff (>)'].values[0]

    retro_plot_data = []

    for b in bins:
        subset = df_valid if b == 'All Data' else df_valid[df_valid['Length_Bin'] == b]
        
        total_baseline = len(subset)
        success_baseline = int(subset['Binding'].sum())
        baseline_rate = (success_baseline / total_baseline * 100) if total_baseline > 0 else 0.0

        passing_retro = subset[subset['dG_ensemble'] > all_data_cutoff]
        n_passing_retro = len(passing_retro)
        n_succ_retro = int(passing_retro['Binding'].sum())
        retro_rate = (n_succ_retro / n_passing_retro * 100) if n_passing_retro > 0 else 0.0

        retro_plot_data.append({
            'Length Bin': b,
            'Type': 'Overall Rate',
            'Success Rate (%)': baseline_rate,
            'Cutoff': None,
            'Success Count': success_baseline,
            'Total Binders': success_baseline
        })
        
        retro_plot_data.append({
            'Length Bin': b,
            'Type': rf'$\Delta$G Cutoff = {all_data_cutoff:.2f}',
            'Success Rate (%)': retro_rate,
            'Cutoff': all_data_cutoff,
            'Success Count': n_succ_retro,
            'Total Binders': success_baseline
        })

    retro_plot_df = pd.DataFrame(retro_plot_data)

    plt.figure(figsize=(11, 6))

    ax = sns.barplot(
        data=retro_plot_df,
        x='Length Bin',
        y='Success Rate (%)',
        hue='Type',
        palette=['#4c72b0', '#e17c05']
    )

    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xlabel('Length Bin', fontsize=12)

    max_rate = retro_plot_df['Success Rate (%)'].max()
    plt.ylim(0, 40)
    plt.legend(title='Metric Threshold', frameon=True, loc='upper right')

    y_row_3 = max_rate + 6
    ax.text(-0.35, y_row_3, "Binders \nRetained:", ha='left', va='center', fontsize=12, fontweight='bold')

    for i, row in retro_plot_df.iterrows():
        bin_idx = bins.index(row['Length Bin'])
        hue_idx = 0 if row['Type'] == 'Overall Rate' else 1
        
        patch_idx = hue_idx * num_bins + bin_idx
        p = ax.patches[patch_idx]
        
        height = p.get_height()
        if not np.isnan(height) and height >= 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', rotation=0, fontsize=9, xytext=(0, 2.5), textcoords='offset points'
            )
            
            if row['Type'] != 'Overall Rate':
                succ = row['Success Count']
                total_binders = row['Total Binders']
                bar_center_x = p.get_x() + p.get_width() / 2.
                ax.text(
                    bar_center_x, y_row_3,
                    f"{succ}/{total_binders}",
                    ha='center', va='center', fontsize=10, fontweight='bold'
                )

    plt.tight_layout()
    plt.savefig('Pareto_retrospective_global_cutoff_comparison.png', dpi=300)
    plt.close()

    # =========================================================================
    # SECTION 4: RETROSPECTIVE BAR PLOT USING USER-DEFINED dG CUTOFF
    # =========================================================================
    user_plot_data = []

    for b in bins:
        subset = df_valid if b == 'All Data' else df_valid[df_valid['Length_Bin'] == b]
        
        total_baseline = len(subset)
        success_baseline = int(subset['Binding'].sum())
        baseline_rate = (success_baseline / total_baseline * 100) if total_baseline > 0 else 0.0

        passing_user = subset[subset['dG_ensemble'] > user_dG_cutoff]
        n_passing_user = len(passing_user)
        n_succ_user = int(passing_user['Binding'].sum())
        user_rate = (n_succ_user / n_passing_user * 100) if n_passing_user > 0 else 0.0

        user_plot_data.append({
            'Length Bin': b,
            'Type': 'Overall Rate',
            'Success Rate (%)': baseline_rate,
            'Cutoff': None,
            'Success Count': success_baseline,
            'Total Binders': success_baseline
        })
        
        user_plot_data.append({
            'Length Bin': b,
            'Type': rf'$\Delta$G Cutoff = {user_dG_cutoff:.2f}',
            'Success Rate (%)': user_rate,
            'Cutoff': user_dG_cutoff,
            'Success Count': n_succ_user,
            'Total Binders': success_baseline
        })

    user_plot_df = pd.DataFrame(user_plot_data)

    plt.figure(figsize=(11, 6))

    ax = sns.barplot(
        data=user_plot_df,
        x='Length Bin',
        y='Success Rate (%)',
        hue='Type',
        palette=['#4c72b0', '#9370db']
    )

    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xlabel('Length Bin', fontsize=12)

    max_rate = user_plot_df['Success Rate (%)'].max()
    plt.ylim(0, 40)
    plt.legend(title='Metric Threshold', frameon=True, loc='upper right')

    y_row_4 = max_rate + 6.1
    ax.text(-0.35, y_row_4, "Binders \nRetained:", ha='left', va='center', fontsize=12, fontweight='bold')

    for i, row in user_plot_df.iterrows():
        bin_idx = bins.index(row['Length Bin'])
        hue_idx = 0 if row['Type'] == 'Overall Rate' else 1
        
        patch_idx = hue_idx * num_bins + bin_idx
        p = ax.patches[patch_idx]
        
        height = p.get_height()
        if not np.isnan(height) and height >= 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', rotation=0, fontsize=9, xytext=(0, 2.5), textcoords='offset points'
            )
            
            if row['Type'] != 'Overall Rate':
                succ = row['Success Count']
                total_binders = row['Total Binders']
                bar_center_x = p.get_x() + p.get_width() / 2.
                ax.text(
                    bar_center_x, y_row_4,
                    f"{succ}/{total_binders}",
                    ha='center', va='center', fontsize=9, fontweight='bold'
                )

    plt.tight_layout()
    plt.savefig('retrospective_user_cutoff_comparison.png', dpi=300)
    plt.close()

    return results_df

if __name__ == '__main__':
    file_path = '/PATH/TO/ESM3DG/DATA.csv'
    find_and_plot_optimized_cutoffs(file_path, min_designs=5, user_dG_cutoff=3.37) #Input desired custom dG cutoff here