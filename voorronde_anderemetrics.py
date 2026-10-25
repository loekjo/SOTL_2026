import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pingouin as pg
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# importing csv into pandas dataframe
# note: remove the headers first before reading the
Anneke_df = pd.read_csv('./Ronde6/2018_AVH.csv', header=0, delimiter=';')
Tom_df = pd.read_csv('./Ronde6/2018_Thom.csv', header=0, delimiter=';')
Loek_df = pd.read_csv('./Ronde6/2018_Loek.csv', header=0, delimiter=';')

def calculate_ac2(scores_df):
    count_matrix, categories = aggregate_raters(scores_df)
    n_subjects, n_categories = count_matrix.shape
    n_raters = 3  # Thom, Loek, Anneke

    # 1. Mathematically Exact Quadratic Weights
    weight_mat = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            weight_mat[i, j] = 1.0 - ((i - j) ** 2 / (n_categories - 1) ** 2)

    # 2. Correct Observed Agreement (Pa) for a Count Matrix
    # For each subject, Gwet's Pa sums (R_ik * R_il * w_kl) for all category pairs,
    # then subtracts self-agreement, and divides by total rater pairs: N * (N - 1)
    pa_total = 0.0
    for row in count_matrix:
        subject_agreement = 0.0
        for k in range(n_categories):
            for l in range(n_categories):
                if k == l:
                    # If same category, subtract the self-pairing of the rater
                    subject_agreement += row[k] * (row[k] - 1) * weight_mat[k, l]
                else:
                    # If different categories, multiply the counts
                    subject_agreement += row[k] * row[l] * weight_mat[k, l]

        # Divide by total possible pairings for this subject: 3 * 2 = 6
        pa_total += subject_agreement / (n_raters * (n_raters - 1))

    pa = pa_total / n_subjects

    # 3. Correct Chance Agreement (Pe)
    # Gwet's Pe uses the overall proportion of classifications (pi) across the whole dataset
    pi = np.sum(count_matrix, axis=0) / (n_subjects * n_raters)
    pe = np.sum(pi * (1.0 - pi)) / (n_categories - 1)

    # 4. Final AC2
    gwet_ac2 = (pa - pe) / (1.0 - pe)
    return gwet_ac2

# 1. Combine the DataFrames
# Note: Ensure the 'SOLO category' columns contain actual numbers (integers/floats)
qf_df = pd.DataFrame({
    'Question': Anneke_df['question#'],
    'Anneke': Anneke_df['Question format'],
    'Thom': Tom_df['Question format'],
    'Loek': Loek_df['Question format']
})
# Note: Ensure the 'SOLO category' columns contain actual numbers (integers/floats)
cd_df = pd.DataFrame({
    'Question': Anneke_df['question#'],
    'Anneke': Anneke_df['Cognitive demand'],
    'Thom': Tom_df['Cognitive demand'],
    'Loek': Loek_df['Cognitive demand']
})

# Note: Ensure the 'SOLO category' columns contain actual numbers (integers/floats)
ct_df = pd.DataFrame({
    'Question': Anneke_df['question#'],
    'Anneke': Anneke_df['Course Theme'],
    'Thom': Tom_df['Course Theme'],
    'Loek': Loek_df['Course Theme']
})

# Select ONLY the columns containing the scores (exclude the 'Question' column)
scores_qf_df = qf_df[['Thom', 'Loek', 'Anneke']]
scores_cd_df = cd_df[['Thom', 'Loek', 'Anneke']]
scores_ct_df = ct_df[['Thom', 'Loek', 'Anneke']]

# 2. Transpose the data (.T) so rows = raters and cols = questions, then convert to a numpy array
reliability_qf = scores_qf_df.T.to_numpy()
count_matrix_qf, categories_qf = aggregate_raters(scores_qf_df)
reliability_cd = scores_cd_df.T.to_numpy()
count_matrix_cd, categories_cd = aggregate_raters(scores_cd_df)
reliability_ct = scores_ct_df.T.to_numpy()
count_matrix_ct, categories_ct = aggregate_raters(scores_ct_df)

# 3. Calculate Krippendorff's alpha
# Use 'ordinal' because SOLO taxonomy has a clear, ranked order.
# If you want to treat the gaps between 1, 2, 3 as perfectly equal math values, use 'interval'.
alpha_qf = krippendorff.alpha(reliability_data=reliability_qf, level_of_measurement='interval')
fleiss_qf = fleiss_kappa(table=count_matrix_qf, method='rand')
ac2_qf = calculate_ac2(scores_qf_df)
alpha_cd = krippendorff.alpha(reliability_data=reliability_cd, level_of_measurement='interval')
fleiss_cd = fleiss_kappa(table=count_matrix_cd, method='rand')
ac2_cd = calculate_ac2(scores_cd_df)
alpha_ct = krippendorff.alpha(reliability_data=reliability_ct, level_of_measurement='interval')
fleiss_ct = fleiss_kappa(table=count_matrix_ct, method='rand')
ac2_ct = calculate_ac2(scores_ct_df)

print(f'reliability scores: \n question format: {alpha_qf} \n cognitive demand: {alpha_cd}\n course theme: {alpha_ct}')

# 2. Melt the DataFrame
qf_df_melted = qf_df.melt(id_vars=['Question'], var_name='Person', value_name='Question format')
cd_df_melted = cd_df.melt(id_vars=['Question'], var_name='Person', value_name='Cognitive demand')
ct_df_melted = ct_df.melt(id_vars=['Question'], var_name='Person', value_name='Course Theme')

# 3. Create manual horizontal offsets for each person
offsets = {'Anneke': -0.1, 'Thom': 0, 'Loek': 0.1}
# Create a new column by adding the offset to the actual Score
qf_df_melted['Offset_Score'] = qf_df_melted.apply(
    lambda row: float(row['Question format']) + offsets[row['Person']], axis=1
)
cd_df_melted['Offset_Score'] = cd_df_melted.apply(
    lambda row: float(row['Cognitive demand']) + offsets[row['Person']], axis=1
)
ct_df_melted['Offset_Score'] = ct_df_melted.apply(
    lambda row: float(row['Course Theme']) + offsets[row['Person']], axis=1
)

# Placeholder variables - Replace these with your actual DataFrames and Alpha scores
dfs = [qf_df_melted, cd_df_melted, ct_df_melted]
alphas = [alpha_qf, alpha_cd, alpha_ct]
fleiss = [fleiss_qf, fleiss_cd, fleiss_ct]
ac2 = [ac2_qf, ac2_cd, ac2_ct]
titles = ['Question format', 'Cognitive demand', 'Course Theme']

# Create a figure with 3 subplots stacked vertically
# Adjust figsize: 18-20 height usually works well for 3 stacked plots
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

for i, ax in enumerate(axes):
    current_df = dfs[i]
    current_alpha = alphas[i]
    current_fleiss = fleiss[i]
    current_ac2 = ac2[i]

    # Create the scatterplot on the specific axis (ax=ax)
    sns.scatterplot(
        data=current_df,
        y='Question',
        x='Offset_Score',
        hue='Person',
        s=80,
        alpha=0.8,
        linewidth=1,
        ax=ax,
        legend=(i == 0)  # Only shows the legend on the first plot to save space
    )

    # Formatting the specific subplot
    ax.set_title(titles[i], fontweight='bold')
    ax.set_ylabel('Question')
    ax.set_xlabel('Solo Category')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add horizontal dividers relative to the questions in THIS dataframe
    # We use range(len(...)) based on the unique questions in the current DF
    unique_questions = current_df['Question'].unique()
    for y_pos in range(len(unique_questions) - 1):
        ax.axhline(y_pos + 0.5, color='gray', linestyle='-', alpha=0.3)

    # Add the Krippendorff Alpha text box to the bottom right of each subplot
    ax.text(
        0.95, 0.05,
        f"Krippendorff's \u03B1: {current_alpha:.3f}\nfleiss: {current_fleiss:.3f}\nGwets AC2: {current_ac2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

# Adjust layout so titles and labels don't overlap
plt.tight_layout()

# Save the combined figure
plt.savefig('./Ronde6/Results_2018_QFCDCT.png', dpi=300)
plt.show()