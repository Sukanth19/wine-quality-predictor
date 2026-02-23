import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')


def load_data():
    red = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'), sep=';')
    white = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')
    red['type'] = 'red'
    white['type'] = 'white'
    combined = pd.concat([red, white], ignore_index=True)
    return red, white, combined


def get_summary_stats(df):
    return df.describe().round(3).to_dict()


def generate_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    red, white, combined = load_data()

    # --- Plot 1: Quality Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax, (df, label, color) in zip(axes, [(red, 'Red Wine', '#c0392b'), (white, 'White Wine', '#f0e68c')]):
        ax.set_facecolor('#16213e')
        counts = df['quality'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{label} Quality Distribution', color='white', fontsize=13)
        ax.set_xlabel('Quality Score', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'quality_dist.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Correlation Heatmap (Red) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    numeric_cols = red.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr().round(2), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, linewidths=0.5,
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    ax.set_title('Red Wine — Feature Correlation', color='white', fontsize=14)
    ax.tick_params(colors='white', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'corr_red.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Correlation Heatmap (White) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    numeric_cols = white.select_dtypes(include=np.number)
    sns.heatmap(numeric_cols.corr().round(2), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax, linewidths=0.5,
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    ax.set_title('White Wine — Feature Correlation', color='white', fontsize=14)
    ax.tick_params(colors='white', labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'corr_white.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Alcohol vs Quality (Boxplot) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax, (df, label, color) in zip(axes, [(red, 'Red Wine', '#c0392b'), (white, 'White Wine', '#f0e68c')]):
        ax.set_facecolor('#16213e')
        groups = [df[df['quality'] == q]['alcohol'].values for q in sorted(df['quality'].unique())]
        bp = ax.boxplot(groups, labels=sorted(df['quality'].unique()),
                        patch_artist=True, medianprops=dict(color='white', linewidth=2))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{label} — Alcohol by Quality', color='white', fontsize=13)
        ax.set_xlabel('Quality Score', color='white')
        ax.set_ylabel('Alcohol (%)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'alcohol_quality.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # --- Plot 5: Feature Histograms (combined) ---
    features = ['fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'alcohol', 'pH', 'sulphates', 'density']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    axes = axes.flatten()
    for ax, feat in zip(axes, features):
        ax.set_facecolor('#16213e')
        ax.hist(red[feat], bins=30, alpha=0.6, color='#c0392b', label='Red', edgecolor='none')
        ax.hist(white[feat], bins=30, alpha=0.6, color='#f0e68c', label='White', edgecolor='none')
        ax.set_title(feat, color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=7)
        ax.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e', edgecolor='none')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    plt.suptitle('Feature Distributions: Red vs White', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_hist.png'), dpi=100, bbox_inches='tight')
    plt.close()


def get_feature_columns():
    return [
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]


def predict_quality(input_values: dict, wine_type: str):
    """
    Rule-based quality estimator using z-scores against the dataset mean/std.
    No ML — pure statistics.
    """
    red, white, _ = load_data()
    df = red if wine_type == 'red' else white
    features = get_feature_columns()

    means = df[features].mean()
    stds = df[features].std()

    # Compute z-scores for the input
    z_scores = {}
    for feat in features:
        val = float(input_values[feat])
        z = (val - means[feat]) / stds[feat] if stds[feat] != 0 else 0
        z_scores[feat] = round(z, 3)

    # Quality-correlated features (positive = higher quality)
    positive_features = ['citric acid', 'sulphates', 'alcohol']
    negative_features = ['volatile acidity', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density']

    score = 0
    for feat in positive_features:
        score += z_scores[feat]
    for feat in negative_features:
        score -= z_scores[feat]

    # Map score to quality range 3–9
    dataset_mean_quality = round(df['quality'].mean(), 2)
    quality_std = df['quality'].std()
    raw_quality = dataset_mean_quality + (score / len(features)) * quality_std
    predicted_quality = int(np.clip(round(raw_quality), 3, 9))

    if predicted_quality <= 4:
        label = 'Poor'
        label_class = 'poor'
    elif predicted_quality <= 5:
        label = 'Average'
        label_class = 'average'
    elif predicted_quality <= 6:
        label = 'Good'
        label_class = 'good'
    else:
        label = 'Excellent'
        label_class = 'excellent'

    # Per-feature breakdown table
    breakdown = []
    for feat in features:
        val = float(input_values[feat])
        z = z_scores[feat]
        status = 'Normal'
        if abs(z) > 2:
            status = 'High' if z > 0 else 'Low'
        elif abs(z) > 1:
            status = 'Slightly High' if z > 0 else 'Slightly Low'
        breakdown.append({
            'feature': feat,
            'your_value': val,
            'dataset_mean': round(means[feat], 3),
            'z_score': z,
            'status': status
        })

    return {
        'quality': predicted_quality,
        'label': label,
        'label_class': label_class,
        'breakdown': breakdown
    }