import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm

# Load the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Reverse coding for the 'Like' column: from "I hate it" (-5) to "I love it" (+5)
# We assume 'Like' is a categorical variable in the original data, represented as -5 to +5.
# Convert 'Like' to numeric
like_map = {
    'I hate it!': -5, '-4': -4, '-3': -3, '-2': -2, '-1': -1,
    '0': 0, '+1': 1, '+2': 2, '+3': 3, '+4': 4, 'I love it!': 5
}
mcdonalds['Like.n'] = 6 - mcdonalds['Like'].replace(like_map)

# Verify transformation
print(mcdonalds['Like.n'].value_counts())

# Step to create the formula for regression
# We'll assume the first 11 columns are the predictor variables, just like in R.
predictor_columns = mcdonalds.columns[:11]  # Replace with the actual column names if different
formula = 'Like.n ~ ' + ' + '.join(predictor_columns)

# Use Patsy to create the design matrices for regression
y, X = patsy.dmatrices(formula, data=mcdonalds, return_type='dataframe')

# Fit an Ordinary Least Squares (OLS) model using statsmodels
model = sm.OLS(y, X)
results = model.fit()

# Display the regression summary
print(results.summary())
