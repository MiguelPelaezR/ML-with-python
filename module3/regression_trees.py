### Train to guess the tip from a taxi ###

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

#print(raw_data.head().T)

# Already split the variables
y = raw_data['tip_amount'].values.astype('float32')
X = raw_data.drop(['tip_amount'], axis=1)

# Normalize feature matrix
X = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### MODEL ###

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=8, random_state=35)

dt_reg.fit(X_train, y_train)

# Evaluate with R^2 and MSE
pred_values = dt_reg.predict(X_test)

#Mean squared error:
mse_score = mean_squared_error(y_test, pred_values)
print('MSE score : {0:.3f}'.format(mse_score))

# R^2:
r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

### EXERCISES:
# Ex 1: What if we change the max_depth to 12? How would the MSE and R^2 be affected?

dt_reg_1 = DecisionTreeRegressor(criterion='squared_error', max_depth=12, random_state=35)

dt_reg_1.fit(X_train, y_train)

# Evaluate with R^2 and MSE
pred_values = dt_reg_1.predict(X_test)

#Mean squared error:
mse_score_1 = mean_squared_error(y_test, pred_values)
print('MSE score with max_depth=12: {0:.3f}'.format(mse_score_1))

# R^2:
r2_score_1 = dt_reg.score(X_test,y_test)
print('R^2 score with max_depth=12: {0:.3f}'.format(r2_score_1))

# Ex 2: Identify the top 3 features with the most effect on the tip_amount.
# print(raw_data.corr().T)
# most are fare_amount, tolls_amount and trip_distance

# more clear:
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
print(abs(correlation_values).sort_values(ascending=False)[:3])


#Ex 3: Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the MSE and R^2 value.
dt_3 = raw_data.drop(['VendorID', 'store_and_fwd_flag', 'payment_type', 'improvement_surcharge'], axis=1)

y_3 = dt_3['tip_amount']
X_3 = dt_3.drop(['tip_amount'], axis=1)

X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_reg_3 = DecisionTreeRegressor(criterion='squared_error', max_depth=8)

dt_reg_3.fit(X_3_train, y_3_train)

pred_values_3 = dt_reg_3.predict(X_3_test)

mse_score_3 = mean_squared_error(y_3_test, pred_values_3)

r2_score_3 = dt_reg_3.score(X_3_test, y_3_test)

print(f"Mean squared error 3: {mse_score_3}")

print(f'R^2 error 3: {r2_score_3}')















