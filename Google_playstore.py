import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Load Datasets
apps_df = pd.read_csv('googleplaystore.csv')
reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')

# Clean Apps Data
apps_df.drop_duplicates(inplace=True)
apps_df.drop(index=10472, inplace=True, errors='ignore')

apps_df = apps_df[apps_df['App'].notnull() & apps_df['Category'].notnull()]
apps_df['Reviews'] = pd.to_numeric(apps_df['Reviews'], errors='coerce')
apps_df['Installs'] = apps_df['Installs'].str.replace('+', '', regex=False).str.replace(',', '', regex=False)
apps_df['Installs'] = pd.to_numeric(apps_df['Installs'], errors='coerce')
apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=False)
apps_df['Price'] = pd.to_numeric(apps_df['Price'], errors='coerce')

# Size Conversion
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', '')) * 1024 * 1024
    elif 'k' in size:
        return float(size.replace('k', '')) * 1024
    elif size == 'Varies with device':
        return np.nan
    return np.nan

apps_df['Size_in_bytes'] = apps_df['Size'].apply(convert_size)
apps_df['Size_MB'] = apps_df['Size_in_bytes'] / (1024 * 1024)

# Drop rows with critical nulls
apps_df.dropna(subset=['Reviews', 'Installs', 'Price'], inplace=True)

# Fill Ratings by Category Mean
apps_df['Rating'] = pd.to_numeric(apps_df['Rating'], errors='coerce')
apps_df['Rating'] = apps_df.groupby('Category')['Rating'].transform(lambda x: x.fillna(x.mean()))

# Create Install Categories
bins = [-1, 0, 10, 1000, 10000, 100000, 1000000, 10000000, 10000000000]
labels = ['no', 'Very low', 'Low', 'Moderate', 'More than moderate', 'High', 'Very High', 'Top Notch']
apps_df['Installs_category'] = pd.cut(apps_df['Installs'], bins=bins, labels=labels)

# ------------------ Visualizations ------------------

# Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(apps_df['Rating'], bins=20, kde=True)
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Count of Apps by Category
plt.figure(figsize=(12, 8))
sns.countplot(y='Category', data=apps_df, order=apps_df['Category'].value_counts().index)
plt.title('Count of Apps by Category')
plt.xlabel('Count')
plt.ylabel('Category')
plt.grid(True)
plt.show()

# Improved Scatter Plot: Installs vs Rating
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Rating', 
    y='Installs', 
    hue='Category', 
    data=apps_df, 
    palette='tab10', 
    alpha=0.6, 
    s=40
)
plt.yscale('log')  # Log scale for better visual separation
plt.title('Relationship between Installs and Ratings (Log Scale)')
plt.xlabel('App Rating')
plt.ylabel('Number of Installs (Log Scale)')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Category')
plt.tight_layout()
plt.show()

# ------------------ Aggregated Insights ------------------

print("Average Rating by Category:")
print(apps_df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10))

print("\nMost Installed Apps:")
print(apps_df[['App', 'Installs']].sort_values(by='Installs', ascending=False).head(10))

print("\nTop Genres:")
if 'Genres' in apps_df.columns:
    print(apps_df['Genres'].value_counts().head(5))
else:
    print("No 'Genres' column found in dataset.")

# ------------------ Review Sentiment Analysis ------------------

# Clean Reviews Data
reviews_df.dropna(subset=['Translated_Review'], inplace=True)
reviews_df.drop_duplicates(inplace=True)

# Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(get_sentiment)
reviews_df['Sentiment_Label'] = reviews_df['Sentiment_Score'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


sentiment_summary = reviews_df.groupby('App')['Sentiment_Label'].value_counts().unstack().fillna(0)
sentiment_summary['Total_Reviews'] = sentiment_summary.sum(axis=1)
sentiment_summary = sentiment_summary.sort_values(by='Total_Reviews', ascending=False)

print("\nApps with Most Sentiment-Labeled Reviews:")
print(sentiment_summary.head(10))

# Merge sentiment data with app data
merged_df = apps_df.merge(sentiment_summary, left_on='App', right_index=True, how='left')

# Fill missing sentiment with 0
for col in ['Positive', 'Negative', 'Neutral']:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(0)

# Boxplot: Positive Reviews vs Install Category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Installs_category', y='Positive', data=merged_df)
plt.title('Positive Reviews by Install Category')
plt.xlabel('Install Category')
plt.ylabel('Count of Positive Reviews')
plt.grid(True)
plt.show()



print("\nConclusion:")
print("1. Family and Game categories dominate the app count.")
print("2. Games and Communication apps have the most installs and reviews.")
print("3. Events and Education apps score the highest in user ratings.")
print("4. Sentiment analysis shows majority of reviews are positive, correlating well with high ratings in top apps.")
