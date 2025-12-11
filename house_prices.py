import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as rfg
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle

df = pd.read_csv("House Price India.csv")
df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1899-12-30')
df['Year'] = df['Date'].dt.year
yearly = df.groupby(['Postal Code','Year'])['Price'].median().reset_index()
yearly['Year'].nunique()

sns.histplot(df['Price'])
plt.show()

df['number of bathrooms'] = df['number of bathrooms'].astype(int)
df['number of floors'] = df['number of floors'].astype(int)
# df.head()

plt.figure(figsize=(20,8))
sns.boxplot(x='Postal Code', y='Price', data=df)
plt.xticks(rotation=45)
plt.title('Price Distribution by Postal Code')
# plt.show()

postal_summary = df.groupby('Postal Code')['Price'].agg(['mean', 'median', 'count']).sort_values('median', ascending=False)
# print(postal_summary)

postal_summary['median'].plot(kind='bar', figsize=(12,6))
plt.ylabel('Median Price')
plt.title('Median Price by Postal Code')
# plt.show()

corr = df[['Price', 'number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 'number of floors', 
           'waterfront present', 'number of views', 'condition of the house', 'grade of the house', 
           'Area of the basement', 'Built Year', 'Renovation Year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation with Price')
# plt.show()
# corr.head()

sns.boxplot(df['Price'])
plt.title('Price Outliers')
# plt.show()

import numpy as np
df['Price_log'] = np.log1p(df['Price'])
sns.histplot(df['Price_log'], bins=50)
plt.title('Log-Transformed Price Distribution')
# plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot(x='Longitude', y='Lattitude', hue='Price', data=df, palette='viridis', size='Price', sizes=(10,200), alpha=0.6)
plt.title('Price by Location')
# plt.show()

df['Price_per_sqft'] = df['Price'] / df['living area']

postal_median = df.groupby('Postal Code')['Price'].median().to_dict()
df['MedianPriceByPostal'] = df['Postal Code'].map(postal_median)

df['Property_Age'] = df['Year'] - df['Built Year']

non_zero_count = (df['Renovation Year'] != 0).sum()
print(f"Number of non-zero values in column A: {non_zero_count}")

df['Effective_Age'] = np.where(df['Renovation Year'] > 0,
                               df['Year'] - df['Renovation Year'],
                               df['Property_Age'])

postal_median = df.groupby('Postal Code')['Price'].median().to_dict()
df['MedianPriceByPostal'] = df['Postal Code'].map(postal_median)

# Simplified annual appreciation
df['Annual_Appreciation'] = (df['Price'] - df['MedianPriceByPostal']) / df['Effective_Age']

df['Annual_Appreciation_pct'] = ((df['Price'] - df['MedianPriceByPostal']) / df['MedianPriceByPostal']) / df['Effective_Age'] * 100

threshold = 5
df['Good investement'] = np.where(df['Annual_Appreciation_pct'] >= threshold, True, False)

#ML model


X = df[['number of bedrooms', 'number of bathrooms', 'living area','grade of the house','Effective_Age']]
y = df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model =rfg(n_estimators=100, random_state=0)

model.fit(X_train, y_train)
y_pred = model.predict(X)
df['Predicted_2016_Price'] = y_pred.copy()


YEARS_AHEAD = 5

df['Predicted_2021_Price'] = df.apply(
    lambda row: row['Predicted_2016_Price'] * (1 + row['Annual_Appreciation_pct']) ** YEARS_AHEAD,
    axis=1
)

pickle.dump(model , open('house-pricing.pkl' , 'wb'))

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)