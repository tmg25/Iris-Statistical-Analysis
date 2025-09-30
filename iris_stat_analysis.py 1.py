# Import libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for plots
sns.set(style="whitegrid")

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 1. Descriptive Statistics
print("Summary Statistics:")
print(iris_df.describe())

# Plot histograms
iris_df.hist(bins=20, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.show()

# Pairplot to show feature relationships
sns.pairplot(iris_df, hue='species')
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# 2. Inferential Statistics

# T-test: Compare sepal length of Setosa and Versicolor
setosa = iris_df[iris_df['species'] == 'setosa']
versicolor = iris_df[iris_df['species'] == 'versicolor']

t_stat, p_val = stats.ttest_ind(setosa['sepal length (cm)'], versicolor['sepal length (cm)'])
print(f"\nT-Test between Setosa and Versicolor Sepal Length:")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

# ANOVA: Compare sepal length across all species
f_stat, p_val_anova = stats.f_oneway(
    iris_df[iris_df['species'] == 'setosa']['sepal length (cm)'],
    iris_df[iris_df['species'] == 'versicolor']['sepal length (cm)'],
    iris_df[iris_df['species'] == 'virginica']['sepal length (cm)']
)
print(f"\nANOVA Test for Sepal Length among all species:")
print(f"F-statistic: {f_stat:.4f}, P-value: {p_val_anova:.4f}")

# 3. Correlation Analysis
correlation_matrix = iris_df.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


