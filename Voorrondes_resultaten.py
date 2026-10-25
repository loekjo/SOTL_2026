import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pingouin as pg
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# importing csv into pandas dataframe
# note: remove the headers first before reading the
rater1_df = pd.read_csv('rater1.csv', header=0, delimiter=';')
rater2 = pd.read_csv('rater2.csv', header=0, delimiter=';')
rater3 = pd.read_csv('rater3.csv', header=0, delimiter=';')
consensus_df = pd.read_csv('./Ronde6/2022_concenus.csv', header=0, delimiter=';')

# 1. Combine the DataFrames
# Note: Ensure the 'SOLO category' columns contain actual numbers (integers/floats)
solo_df = pd.DataFrame({
    'Question': Anneke_df['question#'],
    'rater1: rater1_df['SOLO category'],
    'rater2': rater2_df['SOLO category'],
    'rater3': rater3_df['SOLO category'],
    "consensus": consensus_df["SOLO category"]
})

# Select ONLY the columns containing the scores (exclude the 'Question' column)
scores_df = solo_df[['rater1', 'rater2', 'rater3']]

# 2. Transpose the data (.T) so rows = raters and cols = questions, then convert to a numpy array
reliability_df = scores_df.T.to_numpy()
# prpare data for fleiss kappa calculation
# 1. Transform the raw ratings into a category count matrix
count_matrix, categories = aggregate_raters(scores_df)
n_subjects, n_categories = count_matrix.shape
n_raters = 3  

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

print(f"Gwet's AC2: {gwet_ac2}")

# 3. Calculate Krippendorff's alpha and fleiss kappa
# Use 'ordinal' because SOLO taxonomy has a clear, ranked order.
# If you want to treat the gaps between 1, 2, 3 as perfectly equal math values, use 'interval'.
alpha = krippendorff.alpha(reliability_data=reliability_df, level_of_measurement='ordinal')
fleiss = fleiss_kappa(table=count_matrix, method='rand')
print(f"Krippendorff's Alpha: {alpha}")
print(f"Fleiss's kappa: {fleiss}")

# 2. Melt the DataFrame
Solo_df_melted = solo_df.melt(id_vars=['Question'], var_name='Person', value_name='Solo category')

# 3. Create manual horizontal offsets for each person
offsets = {'Anneke': -0.15, 'Thom': -0.05, 'Loek': 0.05, 'consensus': 0.15}
# offsets = {'Anneke': -0.1, 'Thom': 0, 'Loek': 0.1}
# Create a new column by adding the offset to the actual Score
# Make sure the 'S' is capitalized here!
Solo_df_melted['Offset_Score'] = Solo_df_melted.apply(
    lambda row: float(row['Solo category']) + offsets[row['Person']], axis=1
)

# 4. Create the plot using SCATTERPLOT
plt.figure(figsize=(8, 10))

sns.scatterplot(  
    data=Solo_df_melted,
    y='Question',
    x='Offset_Score',  
    hue='Person',
    s=80,              
    alpha=0.8,
    linewidth=1
)
plt.xlim(0.5, 5.5)
plt.xticks([1, 2, 3, 4, 5])
plt.title('SOLO Category per Question by Person')
plt.xlabel('Solo Category')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add horizontal dividers
for y in range(len(solo_df['Question']) - 1):
    plt.axhline(y + 0.5, color='gray', linestyle='-', alpha=0.3)

# 4. Add the Krippendorff Alpha text box to the plot
ax = plt.gca() # Get the current axes
ax.text(
    0.2, 0.99, # X and Y position (0.95 is far right, 0.05 is near the bottom)
    f"Krippendorff's \u03B1: {alpha:.3f}\nFleiss kappa: {fleiss:.3f}\nGwet's AC2 {gwet_ac2:.3f}", # The text to display (\u03B1 is the Greek alpha symbol)
    transform=ax.transAxes, # This tells Matplotlib to use relative plot coordinates (0 to 1)
    fontsize=12,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9) # Creates the box
)

plt.tight_layout()
plt.savefig('./Ronde6/Results_2022_c.png')
