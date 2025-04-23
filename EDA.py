# EDA_coffee.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
coffee_id = pd.read_csv("coffee_id.csv")
coffee_clean = pd.read_csv("coffee_clean.csv")
coffee = pd.read_csv("coffee.csv")

# -------------------------------
# 1. Basic Info
# -------------------------------

print("------ SCHEMA & HEAD (First 2 Rows) ------\n")
print("📁 coffee_id.csv:")
print(coffee_id.info())
print(coffee_id.head(2), "\n")

print("📁 coffee_clean.csv:")
print(coffee_clean.info())
print(coffee_clean.head(2), "\n")

print("📁 coffee.csv:")
print(coffee.info())
print(coffee.head(2), "\n")

# -------------------------------
# 2. Missing Value Check
# -------------------------------
print("------ MISSING VALUES CHECK ------\n")
print("coffee_id missing values:\n", coffee_id.isnull().sum(), "\n")
print("coffee_clean missing values:\n", coffee_clean.isnull().sum(), "\n")
print("coffee missing values:\n", coffee.isnull().sum(), "\n")

# -------------------------------
# 3. Duplicate Check
# -------------------------------
print("------ DUPLICATES CHECK ------\n")
print("coffee_id duplicates:", coffee_id.duplicated(subset=["slug", "review_date"]).sum())
print("coffee_clean duplicates:", coffee_clean.duplicated().sum())
print("coffee duplicates:", coffee.duplicated().sum(), "\n")

# -------------------------------
# 4. CATEGORICAL VALUE INSIGHTS
# -------------------------------
print("------ UNIQUE VALUES IN CATEGORICAL COLUMNS ------\n")
print("coffee_clean - roast types:\n", coffee_clean["roast types"].value_counts(), "\n")
print("coffee_clean - regions:\n", coffee_clean["regions"].value_counts(), "\n")
print("coffee_clean - type attributes:\n", coffee_clean["type attributes"].value_counts(), "\n")

# -------------------------------
# 5. Visuals: Roast and Region
# -------------------------------
plt.figure(figsize=(8, 4))
sns.countplot(data=coffee_clean, y="roast types", order=coffee_clean["roast types"].value_counts().index)
plt.title("Distribution of Roast Types")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=coffee_clean, y="regions", order=coffee_clean["regions"].value_counts().index)
plt.title("Distribution of Coffee Regions")
plt.tight_layout()
plt.show()
