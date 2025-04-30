# insights_coffee.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("coffee_cleaned_final.csv")

# -------------------------------
# 1. Top-rated Products and Roasters
# -------------------------------
top_products = df.sort_values(by='rating', ascending=False)[['name', 'roaster', 'rating']].drop_duplicates().head(10)
print("\n🌟 Top 10 Rated Products:\n", top_products)

most_reviewed_roasters = df['roaster'].value_counts().head(10)
print("\n🏭 Most Reviewed Roasters:\n", most_reviewed_roasters)

# -------------------------------
# 2. Seasonal Insights
# -------------------------------
seasonal_trend = df.groupby(['review_year', 'review_month'])['rating'].mean().reset_index()
seasonal_trend['date'] = pd.to_datetime(seasonal_trend[['review_year', 'review_month']].assign(day=1))

plt.figure(figsize=(10, 4))
sns.lineplot(data=seasonal_trend, x='date', y='rating')
plt.title("Average Rating Over Time (Seasonal Trend)")
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Regional Performance
# -------------------------------
plt.figure(figsize=(8, 4))
sns.barplot(data=df, y='regions', x='rating', estimator='mean', ci=None, order=df['regions'].value_counts().index)
plt.title("Average Rating by Region")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Roast Preference Analysis
# -------------------------------
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='roast types', y='rating', estimator='mean', ci=None)
plt.xticks(rotation=45)
plt.title("Average Rating by Roast Type")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Correlation Analysis (Sensory Features)
# -------------------------------
# Only include if 'flavor' or 'aroma' were extracted numerically
if 'flavor' in df.columns and pd.api.types.is_numeric_dtype(df['flavor']):
    sns.scatterplot(data=df, x='flavor', y='rating')
    plt.title("Flavor Score vs Rating")
    plt.show()

if 'aroma' in df.columns and pd.api.types.is_numeric_dtype(df['aroma']):
    sns.scatterplot(data=df, x='aroma', y='rating')
    plt.title("Aroma Score vs Rating")
    plt.show()
