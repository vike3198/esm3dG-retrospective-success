import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

# Load the CSV dataset
file_path = "/PATH/TO/ESM3DG/DATA.csv"
df = pd.read_csv(file_path)

# Handle potential column casing variations ('binding' vs 'Binding')
binding_col = "binding" if "binding" in df.columns else "Binding"

# Map 1 -> 'Binder' and 0 -> 'Non-binder'
df["Binding Label"] = df[binding_col].map({1: "Binder", 0: "Non-binder"})


# Categorize sequence length into bins (<80, 80-100, >100)
def categorize_length(Length):
    if Length < 80:
        return "< 80"
    elif 80 <= Length <= 100:
        return "80-100"
    else:
        return "> 100"


df["Length Category"] = df["Length"].apply(categorize_length)

# Global plotting configurations
sns.set_theme(style="whitegrid")
category_order = ["Binder", "Non-binder"]

# Colorblind-safe palette mapping for sources
unique_sources = df["source"].dropna().unique()
cb_palette = sns.color_palette("colorblind", n_colors=len(unique_sources))
source_color_map = dict(zip(unique_sources, cb_palette))

# Create custom legend elements for 'Source'
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=str(src),
        markerfacecolor=color,
        markersize=6,
    )
    for src, color in source_color_map.items()
]

# List to store t-test results
ttest_results = []


def run_two_tailed_ttest(subset_df, group_name, category_type):
    """Calculates Welch's two-tailed t-test between Binders and Non-binders."""
    binders = subset_df[subset_df["Binding Label"] == "Binder"][
        "dG_ensemble"
    ].dropna()
    non_binders = subset_df[subset_df["Binding Label"] == "Non-binder"][
        "dG_ensemble"
    ].dropna()

    n_b = len(binders)
    n_nb = len(non_binders)

    if n_b >= 2 and n_nb >= 2:
        t_stat, p_val = stats.ttest_ind(binders, non_binders, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    res = {
        "Category_Type": category_type,
        "Group_Name": group_name,
        "N_Binders": n_b,
        "N_NonBinders": n_nb,
        "Mean_dG_Binder": binders.mean() if n_b > 0 else np.nan,
        "Mean_dG_NonBinder": non_binders.mean() if n_nb > 0 else np.nan,
        "t_statistic": t_stat,
        "p_value": p_val,
    }
    ttest_results.append(res)
    return res


def annotate_violinplot_pval(ax, x1, x2, p_val, y_max, h_factor=0.03):
    """Draws a bracket with the p-value above a plot pair."""
    if np.isnan(p_val):
        p_str = "n/a"
    elif p_val < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_val:.3f}"

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_range * h_factor
    y = y_max + h

    ax.plot(
        [x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black", clip_on=False
    )
    ax.text(
        (x1 + x2) * 0.5,
        y + h * 1.2,
        p_str,
        ha="center",
        va="bottom",
        color="black",
        fontsize=10,
        fontweight="bold",
    )


def annotate_n_counts(ax, x_pos, y_pos, n_val):
    """Annotates sample size (n=...) directly above a violin."""
    ax.text(
        x_pos,
        y_pos,
        f"n = {n_val}",
        ha="center",
        va="bottom",
        color="black",
        fontsize=9,
        fontweight="bold",
    )


# =========================================================
# 1. OVERALL DATASET VIOLIN PLOT WITH ALL DATA POINTS & T-TEST
# =========================================================
res_overall = run_two_tailed_ttest(df, "Overall Dataset", "Overall")

plt.figure(figsize=(8, 6))

ax = sns.violinplot(
    x="Binding Label",
    y="dG_ensemble",
    data=df,
    order=category_order,
    color="#e0e0e0",
    inner=None,
    cut=0,
    width=0.5,
)

# Overlay points using scatter plot
np.random.seed(42)
x_coords_overall = []
y_coords_overall = []
colors_overall = []

for idx, label in enumerate(category_order):
    sub = df[df["Binding Label"] == label]
    jitter = np.random.uniform(-0.12, 0.12, size=len(sub))
    x_coords_overall.extend(idx + jitter)
    y_coords_overall.extend(sub["dG_ensemble"])
    colors_overall.extend(
        [source_color_map.get(src, "black") for src in sub["source"]]
    )

ax.scatter(
    x_coords_overall,
    y_coords_overall,
    c=colors_overall,
    s=12,
    alpha=0.7,
    zorder=3,
    edgecolors="none",
)

plt.xlabel("")
plt.ylabel(
    "Augmented ESM3dG Predicted $\Delta$G (kcal/mol)", fontsize=12, fontweight="bold"
)

plt.legend(
    handles=legend_elements,
    title="Source",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True,
)

ax.set_xlim(-0.6, 1.6)

# Calculate local max values and add annotations
max_binder = (
    df[df["Binding Label"] == "Binder"]["dG_ensemble"].max()
    if res_overall["N_Binders"] > 0
    else 0
)
max_nonbinder = (
    df[df["Binding Label"] == "Non-binder"]["dG_ensemble"].max()
    if res_overall["N_NonBinders"] > 0
    else 0
)
y_max_overall = max(max_binder, max_nonbinder)

y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
annotate_n_counts(ax, 0, max_binder + y_offset, res_overall["N_Binders"])
annotate_n_counts(ax, 1, max_nonbinder + y_offset, res_overall["N_NonBinders"])

# Place p-value bracket above the highest n-annotation
annotate_violinplot_pval(
    ax, 0, 1, res_overall["p_value"], y_max_overall + (y_offset * 3)
)

plt.tight_layout()
plt.savefig(
    "overall_dG_binder_ensemble_violinplot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# =========================================================
# 2. BINNED BY SEQUENCE LENGTH VIOLIN PLOT WITH ALL DATA POINTS & T-TESTS
# =========================================================
length_order = ["< 80", "80-100", "> 100"]
binding_offsets = {"Binder": -0.20, "Non-binder": 0.20}

plt.figure(figsize=(10, 6))

ax = sns.violinplot(
    x="Length Category",
    y="dG_ensemble",
    hue="Binding Label",
    data=df,
    order=length_order,
    hue_order=category_order,
    palette={"Binder": "#d0d0d0", "Non-binder": "#ececec"},
    inner=None,
    cut=0,
    density_norm="width",
)

# Compute point positioning
np.random.seed(42)
x_coords_len = []
y_coords_len = []
colors_len = []

for idx, length_cat in enumerate(length_order):
    for label in category_order:
        sub_df = df[
            (df["Length Category"] == length_cat)
            & (df["Binding Label"] == label)
        ]
        if sub_df.empty:
            continue

        base_x = idx + binding_offsets[label]
        jitter = np.random.uniform(-0.12, 0.12, size=len(sub_df))

        x_coords_len.extend(base_x + jitter)
        y_coords_len.extend(sub_df["dG_ensemble"])
        colors_len.extend(
            [source_color_map.get(src, "black") for src in sub_df["source"]]
        )

ax.scatter(
    x_coords_len,
    y_coords_len,
    c=colors_len,
    s=12,
    alpha=0.7,
    zorder=3,
    edgecolors="none",
)

plt.legend(
    handles=legend_elements,
    title="Source",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True,
)

ax.set_xlim(-0.6, 2.6)

# Annotate n-counts and p-values for each length bin
for idx, l_cat in enumerate(length_order):
    sub = df[df["Length Category"] == l_cat]
    res_len = run_two_tailed_ttest(sub, l_cat, "Length Bin")

    sub_binder = sub[sub["Binding Label"] == "Binder"]["dG_ensemble"].dropna()
    sub_nonbinder = sub[sub["Binding Label"] == "Non-binder"][
        "dG_ensemble"
    ].dropna()

    max_b = sub_binder.max() if not sub_binder.empty else 0
    max_nb = sub_nonbinder.max() if not sub_nonbinder.empty else 0
    bin_y_max = max(max_b, max_nb)

    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    x_binder = idx - 0.20
    x_nonbinder = idx + 0.20

    # Annotate n values above each violin half
    annotate_n_counts(ax, x_binder, max_b + y_offset, res_len["N_Binders"])
    annotate_n_counts(
        ax, x_nonbinder, max_nb + y_offset, res_len["N_NonBinders"]
    )

    # Annotate p-value bracket above both violins and n-labels
    annotate_violinplot_pval(
        ax, x_binder, x_nonbinder, res_len["p_value"], bin_y_max + (y_offset * 3)
    )

plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
plt.ylabel(
    "Augmented ESM3dG Predicted $\Delta$G (kcal/mol)", fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    "length_binned_dG_binder_ensemble_violinplot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# =========================================================
# 3. INDIVIDUAL VIOLIN PLOTS FOR EACH SOURCE
# =========================================================
output_dir = "individual_source_plots"
os.makedirs(output_dir, exist_ok=True)

for src in unique_sources:
    sub_src = df[df["source"] == src]

    # Calculate t-test statistics for this specific source
    res_src = run_two_tailed_ttest(sub_src, src, "Source Specific")

    plt.figure(figsize=(6, 5))

    ax = sns.violinplot(
        x="Binding Label",
        y="dG_ensemble",
        data=sub_src,
        order=category_order,
        color="#e0e0e0",
        inner=None,
        cut=0,
        width=0.5,
    )

    # Overlay scatter points colored by the source's designated palette color
    np.random.seed(42)
    src_color = source_color_map[src]

    for idx, label in enumerate(category_order):
        sub_label = sub_src[sub_src["Binding Label"] == label]
        if not sub_label.empty:
            jitter = np.random.uniform(-0.12, 0.12, size=len(sub_label))
            ax.scatter(
                idx + jitter,
                sub_label["dG_ensemble"],
                color=src_color,
                s=16,
                alpha=0.8,
                zorder=3,
                edgecolors="none",
            )

    ax.set_xlim(-0.6, 1.6)

    plt.xlabel("")
    plt.ylabel(
        "Augmented ESM3dG Predicted $\Delta$G (kcal/mol)",
        fontsize=11,
        fontweight="bold",
    )
    plt.title(f"Source: {src}", fontsize=13, fontweight="bold", pad=12)

    # Calculate local maxes for annotation positioning
    src_binder_data = sub_src[sub_src["Binding Label"] == "Binder"][
        "dG_ensemble"
    ].dropna()
    src_nonbinder_data = sub_src[sub_src["Binding Label"] == "Non-binder"][
        "dG_ensemble"
    ].dropna()

    max_b = src_binder_data.max() if not src_binder_data.empty else 0
    max_nb = src_nonbinder_data.max() if not src_nonbinder_data.empty else 0
    local_max = max(max_b, max_nb)

    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

    # Annotate n values
    annotate_n_counts(ax, 0, max_b + y_offset, res_src["N_Binders"])
    annotate_n_counts(ax, 1, max_nb + y_offset, res_src["N_NonBinders"])

    # Annotate p-value if both categories exist
    annotate_violinplot_pval(
        ax, 0, 1, res_src["p_value"], local_max + (y_offset * 3)
    )

    plt.tight_layout()

    # Sanitize source name for file saving
    safe_filename = re.sub(r"[^\w\-_\. ]", "_", str(src)).replace(" ", "_")
    plt.savefig(
        os.path.join(output_dir, f"violinplot_source_{safe_filename}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# =========================================================
# 4. EXPORT T-TEST P-VALUES TO CSV
# =========================================================
results_df = pd.DataFrame(ttest_results)
results_df.to_csv("ttest_p_values_binderonly_results.csv", index=False)
print(
    f"Execution complete. Individual source plots saved in '{output_dir}/'. T-test results exported to 'ttest_p_values_binderonly_results.csv'."
)