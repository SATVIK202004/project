import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np


df = pd.read_csv('imdb_top_2000_movies.csv')


df['Votes'] = df['Votes'].str.replace(',', '').astype(float)


df['Gross'] = df['Gross'].replace(r'[\$,]', '', regex=True)  # Remove $ and commas
df['Gross'] = df['Gross'].apply(lambda x: float(x.replace('M', '')) * 1e6 if isinstance(x, str) and 'M' in x else np.nan)


df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')


X = df[['Release Year', 'Duration', 'Metascore', 'Votes', 'Gross']]
y = df['IMDB Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_transformer = SimpleImputer(strategy='mean')


pipeline = Pipeline(steps=[
    ('preprocessor', numeric_transformer),
    ('model', LinearRegression())
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
