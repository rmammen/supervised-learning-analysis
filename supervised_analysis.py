"""
Module 6: Supervised Learning — Predicting Career Alignment Archetypes
INST414 - Data Science Techniques
Building on Module 4 clustering of FiveThirtyEight College Majors dataset

Author: Rmammen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
data = {
    'Major': [
        'Petroleum Engineering', 'Mining And Mineral Engineering',
        'Metallurgical Engineering', 'Naval Architecture And Marine Engineering',
        'Chemical Engineering', 'Nuclear Engineering',
        'Actuarial Science', 'Astronomy And Astrophysics',
        'Mechanical Engineering', 'Electrical Engineering',
        'Computer Engineering', 'Aerospace Engineering',
        'Biomedical Engineering', 'Materials Science',
        'Engineering Mechanics Physics And Science',
        'Biological Engineering', 'Industrial And Manufacturing Engineering',
        'General Engineering', 'Architectural Engineering',
        'Court Reporting', 'Computer Science',
        'Electrical Engineering Technology', 'Mechanical Engineering Related Technologies',
        'Environmental Engineering', 'Industrial Production Technologies',
        'Nursing', 'Finance', 'Accounting', 'Economics',
        'Business Management And Administration',
        'Marketing And Marketing Research', 'Human Resources And Personnel Management',
        'Political Science And Government', 'Criminal Justice And Fire Protection',
        'Psychology', 'Social Work', 'Biology',
        'History', 'English Language And Literature',
        'Sociology', 'Liberal Arts', 'Philosophy And Religious Studies',
        'Communications', 'Journalism', 'Advertising And Public Relations',
        'Fine Arts', 'Music', 'Visual And Performing Arts',
        'Commercial Art And Graphic Design', 'Film Video And Photographic Arts',
        'Art History And Criticism', 'Environmental Science',
        'Physical Fitness Parks Recreation And Leisure',
        'Drama And Theater Arts', 'Composition And Rhetoric',
        'Early Childhood Education', 'Elementary Education',
        'Secondary Teacher Education', 'Special Needs Education',
        'Social Science Or History Teacher Education',
        'Teacher Education: Multiple Levels',
        'Language And Drama Education',
        'General Education', 'Educational Administration And Supervision',
        'Mathematics', 'Statistics And Decision Science', 'Physics'
    ],
    'Major_category': [
        'Engineering','Engineering','Engineering','Engineering',
        'Engineering','Engineering','Business','Physical Sciences',
        'Engineering','Engineering','Engineering','Engineering',
        'Engineering','Engineering','Engineering','Engineering',
        'Engineering','Engineering','Engineering',
        'Law & Public Policy','Computers & Mathematics',
        'Engineering Technologies','Engineering Technologies',
        'Engineering','Engineering Technologies',
        'Health','Business','Business','Social Science',
        'Business','Business','Business',
        'Social Science','Law & Public Policy',
        'Psychology & Social Work','Psychology & Social Work',
        'Biology & Life Science','Humanities & Liberal Arts',
        'Humanities & Liberal Arts','Social Science',
        'Humanities & Liberal Arts','Humanities & Liberal Arts',
        'Communications & Journalism','Communications & Journalism',
        'Communications & Journalism',
        'Arts','Arts','Arts','Arts','Arts','Arts',
        'Biology & Life Science',
        'Industrial Arts & Consumer Services',
        'Arts','Humanities & Liberal Arts',
        'Education','Education','Education','Education',
        'Education','Education','Education','Education','Education',
        'Computers & Mathematics','Computers & Mathematics','Physical Sciences'
    ],
    'Median': [
        110000,75000,73000,70000,65000,65000,
        62000,62000,60000,60000,58000,60000,
        60000,58000,58000,57100,57000,56000,54000,
        54000,53000,52000,50000,50000,46000,
        48000,45000,45000,45000,38000,38000,45000,
        40000,35000,30000,30000,33000,
        32000,32000,30000,30000,30000,
        35000,35000,35000,
        30000,30000,30000,33000,30000,30000,30000,
        30000,28000,31000,
        28000,32000,32000,32000,
        30000,32000,30000,32000,32000,
        45000,45000,45000
    ],
    'Unemployment_rate': [
        0.018,0.071,0.024,0.050,0.061,0.022,
        0.095,0.021,0.057,0.059,0.065,0.044,
        0.048,0.023,0.006,0.029,0.042,0.071,0.059,
        0.116,0.063,0.080,0.057,0.061,0.054,
        0.044,0.060,0.069,0.097,0.072,0.068,0.059,
        0.116,0.097,0.083,0.114,0.060,
        0.082,0.087,0.105,0.099,0.101,
        0.072,0.072,0.072,
        0.161,0.075,0.117,0.097,0.110,0.119,0.096,
        0.095,0.122,0.097,
        0.047,0.052,0.055,0.052,
        0.097,0.044,0.059,0.059,0.071,
        0.072,0.072,0.050
    ],
    'Full_time_rate': [
        0.87,0.78,0.83,0.84,0.86,0.85,
        0.80,0.72,0.87,0.87,0.86,0.86,
        0.81,0.85,0.87,0.86,0.84,0.85,0.83,
        0.79,0.83,0.79,0.78,0.85,0.76,
        0.79,0.81,0.81,0.77,0.71,0.71,0.74,
        0.65,0.73,0.60,0.60,0.65,
        0.59,0.57,0.57,0.54,0.56,
        0.62,0.63,0.62,
        0.51,0.47,0.50,0.59,0.50,0.47,0.62,
        0.59,0.47,0.55,
        0.72,0.76,0.75,0.78,
        0.62,0.77,0.67,0.77,0.75,
        0.71,0.72,0.73
    ],
    'College_job_rate': [
        0.85,0.74,0.77,0.82,0.82,0.80,
        0.77,0.62,0.80,0.81,0.78,0.81,
        0.80,0.79,0.83,0.78,0.75,0.74,0.74,
        0.74,0.74,0.59,0.55,0.72,0.55,
        0.84,0.63,0.66,0.54,0.53,0.54,0.60,
        0.50,0.50,0.38,0.70,0.44,
        0.35,0.44,0.43,0.36,0.39,
        0.50,0.50,0.49,
        0.30,0.32,0.33,0.43,0.36,0.34,0.38,
        0.36,0.36,0.42,
        0.78,0.80,0.76,0.81,
        0.55,0.80,0.67,0.78,0.72,
        0.65,0.63,0.65
    ],
    'Low_wage_rate': [
        0.04,0.06,0.05,0.05,0.05,0.05,
        0.05,0.09,0.05,0.05,0.05,0.05,
        0.05,0.05,0.04,0.05,0.06,0.06,0.06,
        0.06,0.07,0.08,0.09,0.06,0.09,
        0.08,0.10,0.10,0.12,0.14,0.14,0.11,
        0.14,0.13,0.17,0.18,0.15,
        0.18,0.18,0.18,0.20,0.19,
        0.15,0.15,0.16,
        0.23,0.23,0.24,0.19,0.21,0.22,0.18,
        0.18,0.24,0.18,
        0.07,0.07,0.07,0.07,
        0.13,0.07,0.11,0.07,0.08,
        0.10,0.10,0.09
    ]
}

df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# 2. RECREATE MODULE 4 CLUSTERS
# ─────────────────────────────────────────────
features = ['Median','Unemployment_rate','Full_time_rate','College_job_rate','Low_wage_rate']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Name clusters by college_job_rate of centroid
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=features
)
order = centers['College_job_rate'].argsort()[::-1].values
label_names = ['Elite Technical Fields','Credentialed Pipelines',
               'Mixed Outcomes','Misaligned / Precarious']
label_map = {int(cluster_id): name for cluster_id, name in zip(order, label_names)}
df['Cluster_name'] = df['Cluster'].map(label_map)

print("Cluster distribution:")
print(df['Cluster_name'].value_counts())

# ─────────────────────────────────────────────
# 3. KNN — LEAVE-ONE-OUT CROSS VALIDATION
# ─────────────────────────────────────────────
# Use k=5 (odd number avoids ties, large enough for stability)
K = 5
loo = LeaveOneOut()
loo_preds = []
for train_idx, test_idx in loo.split(X_scaled):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_scaled[train_idx], df['Cluster'].values[train_idx])
    loo_preds.append(knn.predict(X_scaled[test_idx])[0])

df['LOO_pred'] = loo_preds
df['LOO_pred_name'] = df['LOO_pred'].map(label_map)
df['Correct'] = df['Cluster'] == df['LOO_pred']

loo_acc = df['Correct'].mean()
print(f"\nLOO Accuracy (k={K}): {loo_acc:.3f}")

# Also scan k values 1-11 using cross_val_score for the k-selection figure
k_vals = range(1, 12)
k_accs = []
y = df['Cluster'].values
for k in k_vals:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_scaled, y, cv=LeaveOneOut(), scoring='accuracy')
    k_accs.append(scores.mean())
    print(f"  k={k}: LOO accuracy = {scores.mean():.3f}")

best_k = list(k_vals)[np.argmax(k_accs)]
print(f"\nBest k = {best_k} (acc={max(k_accs):.3f})")

# ─────────────────────────────────────────────
# 4. MISCLASSIFIED SAMPLES
# ─────────────────────────────────────────────
wrong = df[~df['Correct']].copy()
print(f"\nMisclassified ({len(wrong)} total):")
print(wrong[['Major','Cluster_name','LOO_pred_name',
             'College_job_rate','Median','Unemployment_rate','Low_wage_rate']].to_string())

# Pick top 5 (pad with borderline cases if fewer than 5)
if len(wrong) < 5:
    # add borderline correct cases to illustrate
    df['dist_to_boundary'] = np.abs(
        df['College_job_rate'] - df.groupby('Cluster')['College_job_rate'].transform('mean')
    )
    borderline = df[df['Correct']].nsmallest(5 - len(wrong), 'dist_to_boundary')
    borderline = borderline.copy()
    borderline['LOO_pred_name'] = borderline['Cluster_name']
    borderline['note'] = 'near boundary'
    wrong['note'] = 'misclassified'
    five_cases = pd.concat([wrong, borderline])
else:
    five_cases = wrong.head(5)
    five_cases = five_cases.copy()
    five_cases['note'] = 'misclassified'

print(f"\n5 focus cases:")
print(five_cases[['Major','Cluster_name','LOO_pred_name','note',
                   'College_job_rate','Median','Unemployment_rate']].to_string())

# ─────────────────────────────────────────────
# 5. FULL MODEL METRICS (train on all, evaluate on all — for report table)
# ─────────────────────────────────────────────
knn_full = KNeighborsClassifier(n_neighbors=K)
knn_full.fit(X_scaled, y)
y_pred_full = knn_full.predict(X_scaled)
print("\nFull training classification report:")
target_names_list = [label_map[i] for i in sorted(label_map.keys())]
print(classification_report(y, y_pred_full, target_names=target_names_list))

# ─────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────
palette = {
    'Elite Technical Fields': '#1565C0',
    'Credentialed Pipelines': '#2E7D32',
    'Mixed Outcomes': '#E65100',
    'Misaligned / Precarious': '#B71C1C'
}

# PCA for visualisation
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df['PCA1'] = coords[:,0]
df['PCA2'] = coords[:,1]

# ── Fig 1: k-selection curve ──────────────────
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(list(k_vals), k_accs, 'o-', color='#3F51B5', lw=2, ms=7)
ax.axvline(x=best_k, color='crimson', ls='--', alpha=0.8, label=f'Best k={best_k}')
ax.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax.set_ylabel('LOO Accuracy', fontsize=12)
ax.set_title('KNN: LOO Accuracy vs. k Value', fontsize=14, fontweight='bold')
ax.set_ylim(0.7, 1.02)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig1_k_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ── Fig 2: PCA scatter — true vs LOO predicted ───
fig, axes = plt.subplots(1, 2, figsize=(14,6))
for ax, col, title in zip(
    axes,
    ['Cluster_name','LOO_pred_name'],
    ['True Labels (K-Means from Module 4)','KNN Predicted Labels (LOO, k=5)']
):
    for name, color in palette.items():
        mask = df[col] == name
        ax.scatter(df.loc[mask,'PCA1'], df.loc[mask,'PCA2'],
                   color=color, label=name, s=70, alpha=0.85,
                   edgecolors='white', linewidth=0.4)
    # Circle misclassified
    ax.scatter(df.loc[~df['Correct'],'PCA1'], df.loc[~df['Correct'],'PCA2'],
               s=220, facecolors='none', edgecolors='black', lw=2.0,
               zorder=5, label='Misclassified')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.2)
plt.suptitle('KNN Classification — PCA Projection of Feature Space', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ── Fig 3: Confusion matrix (LOO) ────────────
cm = confusion_matrix(df['Cluster'], df['LOO_pred'])
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_list,
            yticklabels=target_names_list, ax=ax)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'LOO Confusion Matrix — KNN (k={K})\nOverall Accuracy: {loo_acc:.1%}',
             fontsize=12, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3")

# ── Fig 4: Cluster feature heatmap ───────────
summary = df.groupby('Cluster_name')[features].mean()
fig, ax = plt.subplots(figsize=(10,5))
summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
sns.heatmap(summary_norm.T, annot=summary.T.round(2), fmt='.2f',
            cmap='RdYlGn', ax=ax, linewidths=0.5,
            cbar_kws={'label':'Normalized value (0=min, 1=max)'})
ax.set_title('Cluster Feature Profiles (raw values annotated)', fontsize=13, fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ── Fig 5: 5 focus cases ─────────────────────
fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(five_cases))
w = 0.35
majors_short = [m[:22]+'…' if len(m)>22 else m for m in five_cases['Major'].tolist()]
true_colors  = [palette[n] for n in five_cases['Cluster_name']]
pred_colors  = [palette[n] for n in five_cases['LOO_pred_name']]
ax.bar(x - w/2, five_cases['College_job_rate'], width=w,
       color=true_colors, alpha=0.9, edgecolor='black', label='True cluster')
ax.bar(x + w/2, five_cases['College_job_rate'], width=w,
       color=pred_colors, alpha=0.5, edgecolor='black', ls='--', label='Predicted cluster')
ax.set_xticks(x)
ax.set_xticklabels(majors_short, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('College Job Rate')
ax.set_title('5 Focus Cases — True (solid) vs. Predicted (hatched) Cluster\n'
             'Color key: Blue=Elite Technical, Green=Credentialed, Orange=Mixed, Red=Misaligned',
             fontsize=10, fontweight='bold')
ax.set_ylim(0,1)
ax.grid(True, axis='y', alpha=0.3)
patches = [mpatches.Patch(color=c, label=n) for n,c in palette.items()]
ax.legend(handles=patches, fontsize=8, bbox_to_anchor=(1.01,1), loc='upper left')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig5_focus_cases.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ─────────────────────────────────────────────
# 7. PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n=== Cluster Summary ===")
summary2 = df.groupby('Cluster_name')[features].mean().round(3)
summary2['n'] = df['Cluster_name'].value_counts()
print(summary2.to_string())

print(f"\n=== LOO Accuracy: {loo_acc:.3f} ===")
print("\nDone — all figures saved to /mnt/user-data/outputs/")
