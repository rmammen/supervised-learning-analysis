# Predicting Career Alignment Archetypes

This module implements a supervised learning workflow using **K-Nearest Neighbors (KNN)** to predict career alignment archetypes (clusters) for college majors based on economic outcomes. It builds upon a previous unsupervised clustering of the *FiveThirtyEight College Majors* dataset.

## Project Overview
The goal is to classify college majors into one of four archetypes based on features like median salary, unemployment rates, and college-level job placement. The script performs:
1. **Unsupervised Archetype Discovery**: Recreates 4 clusters using K-Means.
2. **Supervised Classification**: Trains a KNN model to predict these labels.
3. **Model Evaluation**: Uses **Leave-One-Out (LOO) Cross-Validation** to assess performance.
4. **Visual Analysis**: Generates PCA projections and feature heatmaps.

## Archetype Definitions
* **Elite Technical Fields**: High salary, high college-job placement, low unemployment.
* **Credentialed Pipelines**: Steady placement (e.g., Nursing, Education), moderate salaries.
* **Mixed Outcomes**: Variable job market stability and mid-range wages.
* **Misaligned / Precarious**: Higher unemployment, lower wages, and lower college-job utilization.

## Methodology
- **Features Used**: `Median`, `Unemployment_rate`, `Full_time_rate`, `College_job_rate`, `Low_wage_rate`.
- **Scaling**: All features are standardized using `StandardScaler`.
- **KNN Configuration**: 
    - Optimized using LOO accuracy across a range of *k* values.
    - Default analysis uses $k=5$.
- **Validation**: Leave-One-Out Cross-Validation (LOO-CV) is used due to the relatively small size of the dataset (67 majors).

## Results
- **LOO Accuracy**: 0.940
- **Confusion Matrix**: High consistency between K-Means labels and KNN predictions, with minor misclassifications occurring primarily at cluster boundaries.

## Key Figures Generated
The script outputs several visualizations to `/mnt/user-data/outputs/`:
* `fig1_k_accuracy.png`: Optimization curve for selecting the best *k*.
* `fig2_pca_scatter.png`: Comparison of True (K-Means) vs. Predicted (KNN) labels in a 2D PCA space.
* `fig3_confusion_matrix.png`: Heatmap showing where the model confuses archetypes.
* `fig4_cluster_heatmap.png`: Characteristic profile of each cluster across all features.
* `fig5_focus_cases.png`: Deep dive into specific majors that were misclassified or borderline.

## Dependencies
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
