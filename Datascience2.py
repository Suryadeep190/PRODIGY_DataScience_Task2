import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("train.csv")

# Display the first few rows
df.head()

# Summary statistics
df.describe()

# Data types and missing values
df.info()

# Check for missing values
df.isnull().sum()

# Example: Fill missing values in 'Age' with median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Example: Drop rows with missing values in 'Embarked'
df.dropna(subset=['Embarked'], inplace=True)

# Check for missing values
df.isnull().sum()

# Example: Fill missing values in 'Age' with median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Example: Drop rows with missing values in 'Embarked'
df.dropna(subset=['Embarked'], inplace=True)

# Example: Relationship between Age and Fare
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Fare', data=df, hue='Survived')
plt.title('Age vs Fare by Survival')
plt.show()

# Example: Count plot of Survived
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Count of Survived')
plt.show()

# Example: Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()
