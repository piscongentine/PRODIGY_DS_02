import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\bhara\Downloads\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

# 1. Basic Information about the Dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Summary:")
print(df.info())
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 2. Handle Missing Values
# Fill missing 'Age' with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows where 'Cabin' is missing (optional for simplicity)
df.drop(columns=['Cabin'], inplace=True)

# Verify no missing values remain
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
# Survival Count
print("\nSurvival Counts:")
print(df['Survived'].value_counts())

# Plot Survival Count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Survival by Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set3')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot of Selected Variables
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived', palette='husl')
plt.show()
