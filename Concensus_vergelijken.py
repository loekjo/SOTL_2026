import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.collections as mcoll
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings

# Suppress minor matplotlib warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# 1. HELPER FUNCTIONS & DATA LOADING
# ---------------------------------------------------------------------------
def loadtests(filelist):
    df_list = {}
    for filepath in filelist:
        try:
            df = pd.read_csv(filepath, header=0, delimiter=';', decimal=',')
            if len(df.columns) <= 1:
                df = pd.read_csv(filepath, header=0, sep=None, engine='python', decimal=',')
        except Exception as e:
            raise RuntimeError(f"Error reading {filepath}: {e}")

        df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '')

        for col in df.columns:
            clean = col.lower().replace(' ', '').replace('_', '').replace('-', '')
            if clean in ['totalpoints', 'totalpoint', 'points', 'totaalpunten']:
                df.rename(columns={col: 'Total points'}, inplace=True)
            elif clean in ['p', 'pvalue', 'pwaarde', 'difficulty', 'moeilijkheid']:
                df.rename(columns={col: 'P'}, inplace=True)

        if 'P' in df.columns:
            df['P'] = pd.to_numeric(df['P'].astype(str).str.replace(',', '.'), errors='coerce')

        if 'Total points' not in df.columns:
            raise KeyError(
                f"\n\n{'=' * 60}\n"
                f"MISSING 'Total points' IN FILE:\n  --> {filepath}\n"
                f"Actual columns found:\n  --> {list(df.columns)}\n"
                f"{'=' * 60}\n"
            )
        df_list[filepath] = df
    return df_list


post_covid_test_filepath = ["test1.csv", "test2.csv", "test3.csv",
                            "test4.csv", "test5.csv"]
pre_covid_test_filepath = ["test6.csv", "test7.csv", "test8.csv"]

pre_covid_test = loadtests(pre_covid_test_filepath)
post_covid_test = loadtests(post_covid_test_filepath)


# ---------------------------------------------------------------------------
# 2. INTERNAL GRAPHING HELPERS
# ---------------------------------------------------------------------------
def _add_group_divider(ax, test_sets, by_file=True):
    if len(test_sets) < 2: return
    n_first_group = len(test_sets[0]) if by_file else 1
    ax.axvline(x=n_first_group - 0.5, color='black', linestyle='--', linewidth=1)


def combine_tests(test_dict):
    return pd.concat(test_dict.values(), ignore_index=True)


def _file_label(filepath):
    if filepath is None: return ''
    name = filepath.split('/')[-1].rsplit('.', 1)[0]
    return name.split('_')[0]


def _label_stacked_bars(ax, plot_df):
    totals = plot_df.sum(axis=1)
    for container, col in zip(ax.containers, plot_df.columns):
        values = plot_df[col].values
        pct = 100 * values / totals.replace(0, 1).values
        labels = [f'{p:.0f}%' if v > 0 else '' for v, p in zip(values, pct)]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=8, color='white', fontweight='bold')


def _format_solo(val):
    if pd.isna(val) or val == '' or str(val).strip().lower() in ('nan', 'none'):
        return 'No score'
    try:
        return f'SOLO {int(float(val))}'
    except (ValueError, TypeError):
        return 'No score'


def _sort_solo_cols(c):
    if c == 'No score': return 999
    try:
        return int(c.replace('SOLO ', ''))
    except ValueError:
        return 998


def _add_pairwise_chi2_brackets(ax, raw_df):
    """Calculates Chi-Square for EVERY pair of tests. Draws staggered brackets for p <= 0.05."""
    valid_tests = [idx for idx in raw_df.index if raw_df.loc[idx].sum() > 0]
    significant_pairs = []

    # Calculate all pairwise p-values
    for t1, t2 in combinations(valid_tests, 2):
        c1 = raw_df.loc[t1].values
        c2 = raw_df.loc[t2].values
        mask = (c1 > 0) | (c2 > 0)
        contingency = np.vstack([c1[mask], c2[mask]])

        if contingency.shape[1] > 1:
            chi2, p, dof, ex = chi2_contingency(contingency)
            if p <= 0.05:
                significant_pairs.append((t1, t2, p))

    if not significant_pairs:
        return

    # Find the maximum height of the bars
    max_height = 0
    for rect in ax.patches:
        max_height = max(max_height, rect.get_y() + rect.get_height())

    y_base = max_height * 1.02
    step = max_height * 0.06
    test_to_x = {test: i for i, test in enumerate(raw_df.index)}

    # Sort pairs by distance on the x-axis so short brackets are drawn at the bottom
    significant_pairs.sort(key=lambda x: abs(test_to_x[x[0]] - test_to_x[x[1]]))

    for i, (t1, t2, p) in enumerate(significant_pairs):
        x1, x2 = test_to_x[t1], test_to_x[t2]
        if x1 > x2: x1, x2 = x2, x1

        sig = '***' if p <= 0.001 else '**' if p <= 0.01 else '*'
        y_bracket = y_base + (i * step)

        ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket + step * 0.2, y_bracket + step * 0.2, y_bracket], lw=1,
                c='black')
        ax.text((x1 + x2) * 0.5, y_bracket + step * 0.2, sig, ha='center', va='bottom', color='black', fontsize=9)

    ax.set_ylim(top=y_base + (len(significant_pairs) * step) + step * 1.5)


# ---------------------------------------------------------------------------
# 3. GRAPHING FUNCTIONS (Page 1)
# ---------------------------------------------------------------------------
def plot_points_per_solo(test_sets, labels=None, normalize=True, by_file=True, ax=None):
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    rows = []
    for label, test_dict in zip(labels, test_sets):
        items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
        for filepath, df in items:
            df = df.copy()
            df['solo_clean'] = df['SOLO category'].apply(_format_solo)
            grouped = df.groupby('solo_clean')['Total points'].sum()
            row = {'Test': f"{label}: {_file_label(filepath)}" if by_file else label}
            row.update(grouped.to_dict())
            rows.append(row)

    plot_df_raw = pd.DataFrame(rows).set_index('Test').fillna(0)
    plot_df_raw = plot_df_raw[sorted(plot_df_raw.columns, key=_sort_solo_cols)]

    if normalize:
        plot_df = 100 * plot_df_raw.div(plot_df_raw.sum(axis=1).replace(0, 1), axis=0)
    else:
        plot_df = plot_df_raw.copy()

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind='bar', stacked=True, ax=ax)
    _label_stacked_bars(ax, plot_df)
    ax.set_title('Points per SOLO category' + (' (normalized)' if normalize else ''), fontweight='bold')
    ax.set_xlabel('Test')
    ax.set_ylabel('% of total points' if normalize else 'Total points')
    ax.legend(title='SOLO Category', fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _add_group_divider(ax, test_sets, by_file)

    # if by_file: _add_pairwise_chi2_brackets(ax, plot_df_raw)
    return ax


def plot_points_by_question_format(test_sets, labels=None, format_names=None, normalize=True, by_file=True, ax=None):
    if format_names is None: format_names = {1: 'Open', 2: 'Closed/MC'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    rows = []
    for label, test_dict in zip(labels, test_sets):
        items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
        for filepath, df in items:
            grouped = df.groupby('Question format')['Total points'].sum()
            row = {'Test': f"{label}: {_file_label(filepath)}" if by_file else label}
            row.update(grouped.to_dict())
            rows.append(row)

    plot_df_raw = pd.DataFrame(rows).set_index('Test').fillna(0)
    plot_df_raw = plot_df_raw.rename(columns=format_names)
    col_order = [format_names[k] for k in sorted(format_names) if format_names[k] in plot_df_raw.columns]
    plot_df_raw = plot_df_raw[col_order]

    if normalize:
        plot_df = 100 * plot_df_raw.div(plot_df_raw.sum(axis=1).replace(0, 1), axis=0)
    else:
        plot_df = plot_df_raw.copy()

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind='bar', stacked=True, ax=ax)
    _label_stacked_bars(ax, plot_df)
    ax.set_title('Points by question format' + (' (normalized)' if normalize else ''), fontweight='bold')
    ax.set_xlabel('Test')
    ax.set_ylabel('% of total points' if normalize else 'Total points')
    ax.legend(title='Question Format', fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _add_group_divider(ax, test_sets, by_file)

    # if by_file: _add_pairwise_chi2_brackets(ax, plot_df_raw)
    return ax


def plot_cognitive_demand_distribution(test_sets, labels=None, demand_names=None, normalize=False, by_file=True,
                                       ax=None):
    if demand_names is None: demand_names = {1: 'Quantitative', 2: 'Qualitative', 3: 'Mixed'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    rows = []
    for label, test_dict in zip(labels, test_sets):
        items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
        for filepath, df in items:
            # Calculate by summing points instead of counting rows
            grouped = df.groupby('Cognitive demand')['Total points'].sum()
            row = {'Test': f"{label}: {_file_label(filepath)}" if by_file else label}
            row.update(grouped.to_dict())
            rows.append(row)

    plot_df_raw = pd.DataFrame(rows).set_index('Test').fillna(0)
    plot_df_raw = plot_df_raw.rename(columns=demand_names)
    col_order = [demand_names[k] for k in sorted(demand_names) if demand_names[k] in plot_df_raw.columns]
    plot_df_raw = plot_df_raw[col_order]

    if normalize:
        plot_df = 100 * plot_df_raw.div(plot_df_raw.sum(axis=1).replace(0, 1), axis=0)
    else:
        plot_df = plot_df_raw.copy()

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind='bar', stacked=True, ax=ax)
    _label_stacked_bars(ax, plot_df)

    # Update titles and labels to reflect points
    ax.set_title('Points per cognitive demand' + (' (normalized)' if normalize else ''), fontweight='bold')
    ax.set_xlabel('Test')
    ax.set_ylabel('% of total points' if normalize else 'Total points')
    ax.legend(title='Demand', fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _add_group_divider(ax, test_sets, by_file)

    # if by_file: _add_pairwise_chi2_brackets(ax, plot_df_raw)
    return ax


def plot_points_by_course_theme(test_sets, labels=None, theme_names=None, normalize=True, by_file=True, ax=None):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    rows = []
    for label, test_dict in zip(labels, test_sets):
        items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
        for filepath, df in items:
            grouped = df.groupby('Course Theme')['Total points'].sum()
            row = {'Test': f"{label}: {_file_label(filepath)}" if by_file else label}
            row.update(grouped.to_dict())
            rows.append(row)

    plot_df_raw = pd.DataFrame(rows).set_index('Test').fillna(0)
    plot_df_raw = plot_df_raw.rename(columns=theme_names)
    col_order = [theme_names[k] for k in sorted(theme_names) if theme_names[k] in plot_df_raw.columns]
    plot_df_raw = plot_df_raw[col_order]

    if normalize:
        plot_df = 100 * plot_df_raw.div(plot_df_raw.sum(axis=1).replace(0, 1), axis=0)
    else:
        plot_df = plot_df_raw.copy()

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind='bar', stacked=True, ax=ax)
    _label_stacked_bars(ax, plot_df)
    ax.set_title('Points by course theme' + (' (normalized)' if normalize else ''), fontweight='bold')
    ax.set_xlabel('Test')
    ax.set_ylabel('% of total points' if normalize else 'Total points')
    ax.legend(title='Course Theme', fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _add_group_divider(ax, test_sets, by_file)

    # if by_file: _add_pairwise_chi2_brackets(ax, plot_df_raw)
    return ax


# ---------------------------------------------------------------------------
# 4. GRAPHING FUNCTIONS (Pages 2-9)
# ---------------------------------------------------------------------------
def plot_p_by_solo_violin(test_sets, labels=None, axes=None):
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    if axes is None: fig, axes = plt.subplots(1, len(test_sets), figsize=(6 * len(test_sets), 5), sharey=True)

    for i, (ax, label, test_dict) in enumerate(zip(axes, labels, test_sets)):
        df = combine_tests(test_dict)
        if 'P' not in df.columns: continue
        solo_order = sorted(df['SOLO category'].dropna().unique())
        sns.violinplot(data=df, x='SOLO category', y='P', order=solo_order, ax=ax, color='tab:blue', cut=0,
                       density_norm='width', inner='stick')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('SOLO category')
        ax.set_ylim(-0.05, 1.05)
        if i > 0: ax.set_ylabel(None)

    axes[0].set_ylabel('P (difficulty)')
    axes[0].figure.suptitle('Question Difficulty (P) by SOLO Category', fontsize=14, fontweight='bold', y=0.98)
    return axes


def plot_p_by_course_theme_violin(test_sets, labels=None, theme_names=None, axes=None):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    if axes is None: fig, axes = plt.subplots(1, len(test_sets), figsize=(6 * len(test_sets), 5), sharey=True)

    theme_order = [theme_names[k] for k in sorted(theme_names)]
    for i, (ax, label, test_dict) in enumerate(zip(axes, labels, test_sets)):
        df = combine_tests(test_dict).copy()
        if 'P' not in df.columns: continue
        df['Theme label'] = df['Course Theme'].map(theme_names)
        sns.violinplot(data=df, x='Theme label', y='P', order=theme_order, ax=ax, color='tab:blue', cut=0,
                       density_norm='width', inner='stick')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Course Theme')
        ax.set_ylim(-0.05, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if i > 0: ax.set_ylabel(None)

    axes[0].set_ylabel('P (difficulty)')
    axes[0].figure.suptitle('Question Difficulty (P) by Course Theme', fontsize=14, fontweight='bold', y=0.98)
    return axes


def plot_solo_by_course_theme(test_sets, labels=None, theme_names=None, axes=None):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    if axes is None: fig, axes = plt.subplots(1, len(test_sets), figsize=(8 * len(test_sets), 5), sharey=True)

    theme_order = [theme_names[k] for k in sorted(theme_names)]
    all_solos = set()
    pooled_dfs = []

    for test_dict in test_sets:
        df = combine_tests(test_dict).copy()
        df['Theme label'] = df['Course Theme'].map(theme_names)
        df['solo_clean'] = df['SOLO category'].apply(_format_solo)
        all_solos.update(df['solo_clean'].unique())
        pooled_dfs.append(df)

    solo_cols = sorted(list(all_solos), key=_sort_solo_cols)

    for ax, label, df in zip(axes, labels, pooled_dfs):
        grouped = df.groupby(['Theme label', 'solo_clean'])['Total points'].sum().unstack('solo_clean').fillna(0)
        grouped = grouped.reindex(index=theme_order, columns=solo_cols, fill_value=0)
        grouped = 100 * grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0)
        grouped.plot(kind='bar', stacked=True, ax=ax, legend=False)
        _label_stacked_bars(ax, grouped)

        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Course Theme')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    axes[0].set_ylabel('% of total points')
    handles, plot_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, plot_labels, title='SOLO Category', fontsize='small', bbox_to_anchor=(1.02, 1),
                    loc='upper left')
    axes[0].figure.suptitle('SOLO Category Breakdown per Course Theme', fontsize=14, fontweight='bold', y=0.98)
    return axes


def plot_p_by_course_theme_violin_per_file(test_sets, labels=None, theme_names=None, ncols=3):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]

    theme_order = [theme_names[k] for k in sorted(theme_names)]
    entries = []
    for label, test_dict in zip(labels, test_sets):
        for filepath, df in test_dict.items():
            if 'P' in df.columns:
                entries.append((f"{label}: {_file_label(filepath)}", df))

    n = len(entries)
    if n == 0: return plt.subplots(1, 1)

    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, (title, df) in zip(axes, entries):
        df = df.copy()
        df['Theme label'] = df['Course Theme'].map(theme_names)
        sns.violinplot(data=df, x='Theme label', y='P', order=theme_order, ax=ax, color='tab:blue', cut=0,
                       density_norm='width', inner='stick')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylim(-0.05, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for ax in axes[n:]: ax.set_visible(False)
    fig.text(0.02, 0.5, 'P (difficulty)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.suptitle('Question Difficulty (P) by Theme (Per File)', fontsize=16, fontweight='bold', y=0.98)
    return fig, axes


def _bin_difficulty(df, easy_threshold=0.8, hard_threshold=0.3):
    bins = [-np.inf, hard_threshold, easy_threshold, np.inf]
    bin_labels = [f'Difficult (P<{hard_threshold})', 'In between', f'Easy (P>{easy_threshold})']
    df = df.copy()
    if 'P' in df.columns:
        df['Difficulty bin'] = pd.cut(df['P'], bins=bins, labels=bin_labels)
    return df, bin_labels


def plot_solo_by_difficulty(test_sets, labels=None, easy_threshold=0.8, hard_threshold=0.3, axes=None):
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    if axes is None:
        fig, axes = plt.subplots(1, len(test_sets), figsize=(7 * len(test_sets), 5), sharey=True)
        if len(test_sets) == 1: axes = [axes]

    for i, (ax, label, test_dict) in enumerate(zip(axes, labels, test_sets)):
        df = combine_tests(test_dict)
        if 'P' not in df.columns: continue

        df, bin_order = _bin_difficulty(df, easy_threshold, hard_threshold)
        df['solo_clean'] = df['SOLO category'].apply(_format_solo)

        grouped = df.groupby(['Difficulty bin', 'solo_clean'], observed=True).size().unstack('solo_clean').fillna(0)
        grouped = grouped.reindex(bin_order)
        grouped = grouped[sorted(grouped.columns, key=_sort_solo_cols)]
        grouped = 100 * grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0)

        grouped.plot(kind='bar', stacked=True, ax=ax, legend=False)
        _label_stacked_bars(ax, grouped)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Difficulty Bin')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        if i > 0: ax.set_ylabel(None)

    axes[0].set_ylabel('% of questions')
    handles, plot_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, plot_labels, title='SOLO Category', fontsize='small', bbox_to_anchor=(1.02, 1),
                    loc='upper left')
    axes[0].figure.suptitle('SOLO Category by Difficulty (Pooled)', fontsize=14, fontweight='bold', y=0.98)
    return axes


def plot_solo_by_difficulty_per_file(test_sets, labels=None, easy_threshold=0.8, hard_threshold=0.3, ncols=3):
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]
    entries = []
    for label, test_dict in zip(labels, test_sets):
        for filepath, df in test_dict.items():
            if 'P' in df.columns:
                entries.append((f"{label}: {_file_label(filepath)}", df))

    n = len(entries)
    if n == 0: return plt.subplots(1, 1)

    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    legend_handles, legend_labels = None, None
    for ax, (title, df) in zip(axes, entries):
        df, bin_order = _bin_difficulty(df, easy_threshold, hard_threshold)
        df['solo_clean'] = df['SOLO category'].apply(_format_solo)

        grouped = df.groupby(['Difficulty bin', 'solo_clean'], observed=True).size().unstack('solo_clean').fillna(0)
        grouped = grouped.reindex(bin_order)
        grouped = grouped[sorted(grouped.columns, key=_sort_solo_cols)]
        grouped = 100 * grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0)

        grouped.plot(kind='bar', stacked=True, ax=ax, legend=False)
        _label_stacked_bars(ax, grouped)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(None)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes[n:]: ax.set_visible(False)
    fig.text(0.04, 0.5, '% of questions', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title='SOLO Category', fontsize='small', bbox_to_anchor=(0.86, 0.5),
                   loc='center left')
    fig.suptitle('SOLO Category by Difficulty (Per File)', fontsize=16, fontweight='bold', y=0.98)
    return fig, axes


def plot_solo_by_difficulty_by_theme(test_sets, labels=None, theme_names=None, easy_threshold=0.8, hard_threshold=0.3,
                                     ncols=3):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]

    pooled = []
    for label, test_dict in zip(labels, test_sets):
        df = combine_tests(test_dict)
        if 'P' in df.columns:
            df, bin_order = _bin_difficulty(df, easy_threshold, hard_threshold)
            df['Test set'] = label
            df['solo_clean'] = df['SOLO category'].apply(_format_solo)
            pooled.append(df)

    if not pooled: return plt.subplots(1, 1)

    all_df = pd.concat(pooled, ignore_index=True)
    all_df['Theme label'] = all_df['Course Theme'].map(theme_names)

    theme_order = [theme_names[k] for k in sorted(theme_names) if theme_names[k] in all_df['Theme label'].unique()]
    n = len(theme_order)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    legend_handles, legend_labels = None, None
    for ax, theme in zip(axes, theme_order):
        sub = all_df[all_df['Theme label'] == theme]
        grouped = sub.groupby(['Test set', 'Difficulty bin', 'solo_clean'], observed=True).size().unstack(
            'solo_clean').fillna(0)
        grouped = grouped[sorted(grouped.columns, key=_sort_solo_cols)]
        grouped = 100 * grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0).fillna(0)
        grouped.index = [f"{t}: {b}" for t, b in grouped.index]

        grouped.plot(kind='bar', stacked=True, ax=ax, legend=False)
        _label_stacked_bars(ax, grouped)

        ax.set_title(theme, fontsize=11, fontweight='bold')
        ax.set_xlabel(None)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes[n:]: ax.set_visible(False)
    fig.text(0.04, 0.5, '% of questions', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title='SOLO Category', fontsize='small', bbox_to_anchor=(0.86, 0.5),
                   loc='center left')
    fig.suptitle('SOLO Category by Difficulty (Split by Theme)', fontsize=16, fontweight='bold', y=0.98)
    return fig, axes


def plot_p_by_course_theme_raincloud_per_theme(test_sets, labels=None, theme_names=None, ncols=3):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]

    rows = []
    file_order = []
    for label, test_dict in zip(labels, test_sets):
        for filepath, df in test_dict.items():
            if 'P' in df.columns:
                file_label = f"{label}: {_file_label(filepath)}"
                file_order.append(file_label)
                df = df.copy()
                df['File'] = file_label
                rows.append(df)

    if not rows: return plt.subplots(1, 1)

    all_df = pd.concat(rows, ignore_index=True)
    all_df['Theme label'] = all_df['Course Theme'].map(theme_names)

    theme_order = [theme_names[k] for k in sorted(theme_names) if theme_names[k] in all_df['Theme label'].unique()]
    n = len(theme_order)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for ax, theme in zip(axes, theme_order):
        sub = all_df[all_df['Theme label'] == theme]

        # 1. The Cloud (Violin Plot)
        sns.violinplot(data=sub, x='File', y='P', order=file_order, ax=ax,
                       color='tab:blue', cut=0, inner=None, linewidth=1, #density_norm='width'
                       )

        # 2. The Umbrella (Centered Boxplot)
        sns.boxplot(data=sub, x='File', y='P', order=file_order, ax=ax,
                    width=0.1, color='white', showfliers=False, zorder=2)

        # 3. The Rain (Strip Plot)
        sns.stripplot(data=sub, x='File', y='P', order=file_order, ax=ax,
                      color='black', alpha=0.5, jitter=0.08, size=3.5, zorder=3)

        # --- RAINCLOUD STRUCTURAL HACK ---

        # A. Cut the violins in half so they only span the left side
        for obj in ax.findobj(mcoll.PolyCollection):
            obj.set_alpha(0.5)  # Make the cloud slightly transparent
            for path in obj.get_paths():
                m = np.mean(path.vertices[:, 0])  # Find the exact center of the violin
                # Clip the polygon vertices so nothing extends past the center to the right
                path.vertices[:, 0] = np.clip(path.vertices[:, 0], -np.inf, m)

        # B. Shift the rain dots to the right side
        for obj in ax.findobj(mcoll.PathCollection):
            offsets = obj.get_offsets()
            if len(offsets) > 0:
                # Add 0.15 to the x-coordinates to push the dots right of the boxplot
                offsets[:, 0] += 0.15
                obj.set_offsets(offsets)

        # Formatting
        ax.set_title(theme, fontsize=11, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylim(-0.05, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    for ax in axes[n:]: ax.set_visible(False)
    fig.text(0.04, 0.5, 'P (difficulty)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.suptitle('Question Difficulty (P) Across Files (Per Theme)', fontsize=16, fontweight='bold', y=0.98)
    return fig, axes


def plot_solo_by_test_per_theme_bar(test_sets, labels=None, theme_names=None, by_file=True, ncols=3):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]

    rows = []
    test_order = []
    for label, test_dict in zip(labels, test_sets):
        items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
        for filepath, df in items:
            test_label = f"{label}: {_file_label(filepath)}" if by_file else label
            test_order.append(test_label)
            df = df.copy()
            df['Test'] = test_label
            df['solo_clean'] = df['SOLO category'].apply(_format_solo)
            rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_df['Theme label'] = all_df['Course Theme'].map(theme_names)

    theme_order = [theme_names[k] for k in sorted(theme_names) if theme_names[k] in all_df['Theme label'].unique()]
    n = len(theme_order)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    legend_handles, legend_labels = None, None
    for ax, theme in zip(axes, theme_order):
        sub = all_df[all_df['Theme label'] == theme]
        grouped = sub.groupby(['Test', 'solo_clean'])['Total points'].sum().unstack('solo_clean').fillna(0)
        grouped = grouped.reindex(test_order).dropna(how='all')
        grouped = grouped[sorted(grouped.columns, key=_sort_solo_cols)]
        grouped = 100 * grouped.div(grouped.sum(axis=1).replace(0, 1), axis=0)

        grouped.plot(kind='bar', stacked=True, ax=ax, legend=False)
        _label_stacked_bars(ax, grouped)
        ax.set_title(theme, fontsize=11, fontweight='bold')
        ax.set_xlabel(None)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes[n:]: ax.set_visible(False)
    fig.text(0.04, 0.5, '% of total points', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    if legend_handles:
        fig.legend(legend_handles, legend_labels, title='SOLO Category', fontsize='small', bbox_to_anchor=(0.86, 0.5),
                   loc='center left')
    fig.suptitle('SOLO Category by File (Per Course Theme)', fontsize=16, fontweight='bold', y=0.98)
    return fig, axes


# ---------------------------------------------------------------------------
# 5. STATISTICS & CHI-SQUARE TABLES
# ---------------------------------------------------------------------------
def compute_summary_metrics(test_sets, labels=None, by_file=True, theme_names=None, demand_names=None):
    if theme_names is None: theme_names = {1: 'Calculation', 2: 'Absorption', 3: 'Distribution', 4: 'Hepatic clearance',
                                           5: 'Renal clearance', 6: 'Bioanalysis', 7: 'Variation'}
    if demand_names is None: demand_names = {1: 'Quantitative', 2: 'Qualitative', 3: 'Mixed'}
    if labels is None: labels = [f"Set {i + 1}" for i in range(len(test_sets))]

    def _row(df):
        row = {
            'N questions': len(df),
            'Total points': df['Total points'].sum() if 'Total points' in df.columns else 0,
            'Mean points/question': df['Total points'].mean() if 'Total points' in df.columns else np.nan,
            'Std points/question': df['Total points'].std() if 'Total points' in df.columns else np.nan,
            'Mean cognitive demand': df['Cognitive demand'].mean() if 'Cognitive demand' in df.columns else np.nan,
            'Mean SOLO category': df['SOLO category'].mean() if 'SOLO category' in df.columns else np.nan,
            'Std SOLO category': df['SOLO category'].std() if 'SOLO category' in df.columns else np.nan,
        }
        if 'P' in df.columns:
            p_series = pd.to_numeric(df['P'].astype(str).str.replace(',', '.'), errors='coerce')
            row.update({
                'Mean P': p_series.mean(),
                'Median P': p_series.median(),
                'Std P': p_series.std(),
                'Min P': p_series.min(),
                'Max P': p_series.max(),
                '% P < 0.3 (hard)': 100 * (p_series < 0.3).mean(),
                '% P > 0.8 (easy)': 100 * (p_series > 0.8).mean(),
            })
        else:
            for k in ['Mean P', 'Median P', 'Std P', 'Min P', 'Max P', '% P < 0.3 (hard)', '% P > 0.8 (easy)']: row[
                k] = np.nan

        theme_counts = df['Course Theme'].value_counts() if 'Course Theme' in df.columns else pd.Series()
        for code, name in theme_names.items(): row[f'N ({name})'] = theme_counts.get(code, 0)

        solo_counts = df['SOLO category'].value_counts() if 'SOLO category' in df.columns else pd.Series()
        for solo in sorted(df['SOLO category'].dropna().unique()) if 'SOLO category' in df.columns else []:
            row[f'N (SOLO {int(solo)})'] = solo_counts.get(solo, 0)
        return row

    rows = []
    for label, test_dict in zip(labels, test_sets):
        if by_file:
            for filepath, df in test_dict.items():
                row = {'Test set': label, 'File': _file_label(filepath)}
                row.update(_row(df))
                rows.append(row)
        else:
            df = combine_tests(test_dict)
            row = {'Test set': label, 'File': None}
            row.update(_row(df))
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def generate_chi2_tables_pdf(pdf, test_sets, labels):
    """Calculates pair-wise and pooled Chi-Squared p-values and saves them directly to the PDF."""

    def get_raw_df(metric_code, by_file):
        rows = []
        for label, test_dict in zip(labels, test_sets):
            items = test_dict.items() if by_file else [(None, combine_tests(test_dict))]
            for filepath, df in items:
                if metric_code == 'solo':
                    grouped = df.copy()
                    grouped['solo_clean'] = grouped['SOLO category'].apply(_format_solo)
                    grouped = grouped.groupby('solo_clean')['Total points'].sum()
                elif metric_code == 'format':
                    grouped = df.groupby('Question format')['Total points'].sum()
                elif metric_code == 'demand':
                    grouped = df.groupby('Cognitive demand').size()
                elif metric_code == 'theme':
                    grouped = df.groupby('Course Theme')['Total points'].sum()
                row = {'Test': f"{label}: {_file_label(filepath)}" if by_file else label}
                row.update(grouped.to_dict())
                rows.append(row)
        return pd.DataFrame(rows).set_index('Test').fillna(0)

    metrics = [
        ('SOLO Category', 'solo'), ('Question Format', 'format'),
        ('Cognitive Demand', 'demand'), ('Course Theme', 'theme')
    ]

    # Page 1: Pairwise Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (nice_name, metric_code) in enumerate(metrics):
        df_raw = get_raw_df(metric_code, by_file=True)
        tests = df_raw.index.tolist()
        matrix = pd.DataFrame(index=tests, columns=tests)

        for t1 in tests:
            for t2 in tests:
                if t1 == t2:
                    matrix.loc[t1, t2] = "-"
                else:
                    contingency = np.vstack([df_raw.loc[t1].values, df_raw.loc[t2].values])
                    contingency = contingency[:, contingency.sum(axis=0) > 0]  # drop empty categories
                    if contingency.shape[1] > 1:
                        _, p, _, _ = chi2_contingency(contingency)
                        sig = '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else 'ns'
                        matrix.loc[t1, t2] = f"{p:.3f} {sig}".strip()
                    else:
                        matrix.loc[t1, t2] = "N/A"

        ax = axes[idx]
        ax.axis('off')
        ax.set_title(f'{nice_name} (Pairwise p-values)', fontsize=14, fontweight='bold', pad=10)
        table = ax.table(cellText=matrix.values, rowLabels=matrix.index, colLabels=matrix.columns, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

    fig.suptitle('Pairwise Chi-Squared Tests (Every Test vs Every Test)', fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: Pooled Tests (Paper vs Digital)
    pooled_results = []
    for nice_name, metric_code in metrics:
        df_pool = get_raw_df(metric_code, by_file=False)
        valid = df_pool.loc[(df_pool.sum(axis=1) > 0), (df_pool.sum(axis=0) > 0)]
        if valid.shape[0] > 1 and valid.shape[1] > 1:
            _, p, _, _ = chi2_contingency(valid)
            sig = '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.05 else 'ns'
            pooled_results.append([nice_name, f"{p:.4f}", sig])
        else:
            pooled_results.append([nice_name, "N/A", "N/A"])

    df_pooled_res = pd.DataFrame(pooled_results, columns=['Metric', 'p-value', 'Significance'])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.axis('off')
    ax2.set_title('Pooled Chi-Squared Tests (Paper Pencil vs Digital)', fontsize=16, fontweight='bold', pad=20)
    table2 = ax2.table(cellText=df_pooled_res.values, colLabels=df_pooled_res.columns, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1, 2)
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)


# ---------------------------------------------------------------------------
# 6. EXPORT ALL TO PDF
# ---------------------------------------------------------------------------
test_sets = [pre_covid_test, post_covid_test]
labels = ['Paper Pencil', 'Digital']

with PdfPages('concensus_vergelijken.pdf') as pdf:
    # Page 1: Overview Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_points_per_solo(test_sets, labels, ax=axes[0, 0])
    plot_points_by_question_format(test_sets, labels, ax=axes[0, 1])
    plot_cognitive_demand_distribution(test_sets, labels, ax=axes[1, 0], normalize=True)
    plot_points_by_course_theme(test_sets, labels, ax=axes[1, 1])
    fig.suptitle('Test Overview Comparisons', fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: P by SOLO category
    fig_solo, axes_solo = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plot_p_by_solo_violin(test_sets, labels, axes=axes_solo)
    fig_solo.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig_solo)
    plt.close(fig_solo)

    # Page 3: SOLO category breakdown per Course Theme
    fig_solo_theme, axes_solo_theme = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_solo_by_course_theme(test_sets, labels, axes=axes_solo_theme)
    fig_solo_theme.tight_layout(rect=[0, 0, 0.88, 0.92])
    pdf.savefig(fig_solo_theme)
    plt.close(fig_solo_theme)

    # Page 4: SOLO category (% of points) by test per Course Theme (Grid)
    fig_solo_bar, _ = plot_solo_by_test_per_theme_bar(test_sets, labels)
    fig_solo_bar.tight_layout(rect=[0.04, 0, 0.85, 0.94])
    pdf.savefig(fig_solo_bar)
    plt.close(fig_solo_bar)

    # Page 5: P by Course Theme, per individual test file (Grid Violin)
    fig_theme_files, _ = plot_p_by_course_theme_violin_per_file(test_sets, labels)
    fig_theme_files.tight_layout(rect=[0.04, 0, 1, 0.94])
    pdf.savefig(fig_theme_files)
    plt.close(fig_theme_files)

    # Page 6: P by test file, one subplot per Course Theme (Grid Violin)
    fig_theme_violin_split, _ = plot_p_by_course_theme_raincloud_per_theme(test_sets, labels)
    fig_theme_violin_split.tight_layout(rect=[0.04, 0, 1, 0.94])
    pdf.savefig(fig_theme_violin_split)
    plt.close(fig_theme_violin_split)

    # Page 7: SOLO category by difficulty bin, pooled per test set
    fig_diff, axes_diff = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_solo_by_difficulty(test_sets, labels, axes=axes_diff)
    fig_diff.tight_layout(rect=[0, 0, 0.88, 0.92])
    pdf.savefig(fig_diff)
    plt.close(fig_diff)

    # Page 8: SOLO category by difficulty bin, per individual test file (Grid)
    fig_diff_files, _ = plot_solo_by_difficulty_per_file(test_sets, labels)
    fig_diff_files.tight_layout(rect=[0.04, 0, 0.85, 0.94])
    pdf.savefig(fig_diff_files)
    plt.close(fig_diff_files)

    # Page 9: SOLO by difficulty, one subplot per Course Theme (Grid)
    fig_diff_theme, _ = plot_solo_by_difficulty_by_theme(test_sets, labels)
    fig_diff_theme.tight_layout(rect=[0.04, 0, 0.85, 0.94])
    pdf.savefig(fig_diff_theme)
    plt.close(fig_diff_theme)

    # Page 10: Pooled Statistics Table
    metrics = compute_summary_metrics(test_sets, labels, by_file=False)
    metrics_display = metrics.round(2)
    col_headers = metrics_display['Test set'].tolist()
    stat_columns = [c for c in metrics_display.columns if c not in ('Test set', 'File')]
    table_data = metrics_display[stat_columns].T.values
    n_rows = len(stat_columns)
    n_cols = len(col_headers)

    fig2, ax2 = plt.subplots(figsize=(min(2.5 * n_cols + 4, 16), 0.4 * n_rows + 1.5))
    ax2.axis('off')
    ax2.set_title('Pooled Summary Statistics', fontsize=18, fontweight='bold', pad=20)
    table = ax2.table(cellText=table_data, rowLabels=stat_columns, colLabels=col_headers, loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)

    # Page 11 & 12: Chi-Squared Tables
    generate_chi2_tables_pdf(pdf, test_sets, labels)

print(f"Saved all beautifully formatted plots and statistics to concensus_vergelijken.pdf")
