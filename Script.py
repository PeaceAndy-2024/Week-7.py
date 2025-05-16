# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    display(iris_df.head())
    
    # Explore dataset structure
    print("\nDataset information:")
    iris_df.info()
    
    # Check for missing values
    print("\nMissing values per column:")
    print(iris_df.isnull().sum())
    
    # Clean the dataset (though iris dataset is already clean)
    # For demonstration, we'll show how we would handle missing values
    if iris_df.isnull().sum().sum() > 0:
        iris_df.fillna(iris_df.mean(), inplace=True)  # Fill numerical missing values with mean
        print("\nMissing values after cleaning:")
        print(iris_df.isnull().sum())
    else:
        print("\nNo missing values found - dataset is clean!")
        
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic statistics for numerical columns:")
display(iris_df.describe())

# Group by species and compute mean
print("\nMean measurements by species:")
species_stats = iris_df.groupby('species').mean()
display(species_stats)

# Interesting findings
print("\nInteresting findings:")
print("- Setosa has significantly smaller petal dimensions compared to other species")
print("- Virginica has the largest measurements across all features")
print("- Versicolor is intermediate between setosa and virginica in all measurements")

# Task 3: Data Visualization

# Set style for better looking plots
plt.style.use('seaborn')

# Create figure with subplots
plt.figure(figsize=(15, 12))

# 1. Line chart (showing trends across measurements by species)
plt.subplot(2, 2, 1)
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species].mean(numeric_only=True)
    plt.plot(species_data.index[:4], species_data.values[:4], label=species, marker='o')
plt.title('Average Measurements by Iris Species')
plt.xlabel('Measurement Type')
plt.ylabel('Centimeters (cm)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# 2. Bar chart (average sepal length by species)
plt.subplot(2, 2, 2)
species_stats['sepal length (cm)'].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y')

# 3. Histogram (distribution of petal length)
plt.subplot(2, 2, 3)
iris_df['petal length (cm)'].hist(bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(False)

# 4. Scatter plot (sepal length vs petal length colored by species)
plt.subplot(2, 2, 4)
colors = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}
for species, color in colors.items():
    subset = iris_df[iris_df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                label=species, color=color, alpha=0.7)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Additional observations
print("\nVisualization Observations:")
print("1. Line chart clearly shows setosa is smallest, virginica largest across all measurements")
print("2. Bar chart confirms setosa has shortest average sepal length")
print("3. Histogram reveals petal length has a bimodal distribution")
print("4. Scatter plot shows clear separation between species, especially setosa")
