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

# Create dominant roast column from one-hot encoding
roast_columns = [
    "roast_dark", "roast_light", "roast_medium", 
    "roast_medium_dark", "roast_medium_light", 
    "roast_very_dark", "roast_nan"
]
coffee_clean["dominant_roast"] = coffee_clean[roast_columns].idxmax(axis=1)
print("coffee_clean - roast types:\n", coffee_clean["dominant_roast"].value_counts(), "\n")

# Extract region categories
region_columns = [
    "region_africa_arabia", "region_caribbean", "region_central_america", 
    "region_hawaii", "region_asia_pacific", "region_south_america"
]
coffee_clean["dominant_region"] = coffee_clean[region_columns].idxmax(axis=1)
print("coffee_clean - regions:\n", coffee_clean["dominant_region"].value_counts(), "\n")

# Extract type attribute categories
type_columns = [
    "type_espresso", "type_organic", "type_fair_trade",
    "type_decaffeinated", "type_pod_capsule", "type_blend", "type_estate"
]
coffee_clean["dominant_type"] = coffee_clean[type_columns].idxmax(axis=1)
print("coffee_clean - type attributes:\n", coffee_clean["dominant_type"].value_counts(), "\n")

# -------------------------------
# 5. Visuals: Roast and Region
# -------------------------------
plt.figure(figsize=(8, 4))
sns.countplot(data=coffee_clean, y="dominant_roast", order=coffee_clean["dominant_roast"].value_counts().index)
plt.title("Distribution of Roast Types")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=coffee_clean, y="dominant_region", order=coffee_clean["dominant_region"].value_counts().index)
plt.title("Distribution of Coffee Regions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=coffee_clean, y="dominant_type", order=coffee_clean["dominant_type"].value_counts().index)
plt.title("Distribution of Coffee Type Attributes")
plt.tight_layout()
plt.show()
