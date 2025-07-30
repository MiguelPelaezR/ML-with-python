## OBESITY RISK PREDICTION
# This module implements a classification model to predict obesity risk based on various health metrics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Load the dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

#Visualize the distribution of the target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

# Exercise 1: Check for null values
print("Checking for null values in the dataset:")
print(data.isnull().sum())

print("Data information:")
print(data.info())
print(data.describe())

## Preprecessing data

#Standardize the numerical features
continuos_colums = data.select_dtypes(include = ['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuos_colums])

#convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuos_colums))

# Combine the scaled features with the original categorical features
scaled_data = pd.concat([data.drop(columns=continuos_colums), scaled_df], axis=1)

# One-hot encode the categorical features (convert categorical variables into numerical format)
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude the target variable

#Apply OneHotEncoder to the categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Convert the encoded features back to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combine with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)


# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
print(prepped_data.head())

# Separate features and target variable
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']


### Model Training and Evaluation ##
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Initialize the OneVsOneClassifier with Logistic Regression
'''
In the One-vs-All approach:

The algorithm trains a single binary classifier for each class.
Each classifier learns to distinguish a single class from all the others combined.
If there are k classes, k classifiers are trained.
During prediction, the algorithm evaluates all classifiers on each input, and selects 
the class with the highest confidence score as the predicted class.
'''

model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ova = model_ova.predict(X_test)

# Calculate accuracy
print("One-vs-All (OvA) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


# Initialize the OneVsOneClassifier with Logistic Regression
'''In the One-vs-One approach:
The algorithm trains a binary classifier for every pair of classes.
For k classes, k*(k-1)/2 classifiers are trained.
Each classifier learns to distinguish between two classes.
During prediction, each classifier votes for one of the two classes it was trained on.
The class with the most votes is selected as the predicted class.
'''

# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ovo = model_ovo.predict(X_test)
# Calculate accuracy
print("One-vs-One (OvO) Strategy:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


# Visualize the results
plt.figure(figsize=(10, 5))
sns.countplot(x=y_test, hue=y_pred_ovo, palette='Set1')
plt.title('Predicted vs Actual Obesity Levels')
plt.xlabel('Actual Obesity Level')
plt.ylabel('Count')
plt.legend(title='Predicted Obesity Level', loc='upper right')
plt.show()

# EXERCISES:
# Exercise 1: Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.
print ("\nVisualizing results with test size 0.1:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# OneVsAllClassifier with Logistic Regression test = 0.1
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)
print("One-vs-All (OvA) Strategy with 0.1:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# OneVsOneClassifier with Logistic Regression test = 0.1
model_ovo.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ovo = model_ovo.predict(X_test)
# Calculate accuracy
print("One-vs-One (OvO) Strategy with 0.1:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

print ("\nVisualizing results with test size 0.3:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# OneVsAllClassifier with Logistic Regression test = 0.3
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)
print("One-vs-All (OvA) Strategy with 0.3:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# OneVsOneClassifier with Logistic Regression test = 0.3
model_ovo.fit(X_train, y_train)

# Make predictions on the test set
y_pred_ovo = model_ovo.predict(X_test)
# Calculate accuracy
print("One-vs-One (OvO) Strategy with 0.3:")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

'''
Esto está mejor hecho
for test_size in [0.1, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model_ova.fit(X_train, y_train)
    y_pred = model_ova.predict(X_test)
    print(f"Test Size: {test_size}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
'''

# Exercise 2: Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model.
# Feature importance
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# For One vs One model
# Collect all coefficients from each underlying binary classifier
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])

# Now take the mean across all those classifiers
feature_importance = np.mean(np.abs(coefs), axis=0)

# Plot feature importance
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance (One-vs-One)")
plt.xlabel("Importance")
plt.show()

# Exercise 3: Write a function obesity_risk_pipeline to automate the entire pipeline
def obesity_risk_pipeline(data_path, test_size=0.2):
    # Load the dataset
    data = pd.read_csv(data_path)
    ## Preprecessing data

    #Standardize the numerical features
    continuos_colums = data.select_dtypes(include = ['float64']).columns.tolist()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuos_colums])

    #convert the scaled features back to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuos_colums))

    # Combine the scaled features with the original categorical features
    scaled_data = pd.concat([data.drop(columns=continuos_colums), scaled_df], axis=1)

    # One-hot encode the categorical features (convert categorical variables into numerical format)
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude the target variable

    #Apply OneHotEncoder to the categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

    # Convert the encoded features back to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)


    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
    print(prepped_data.head())

    # Separate features and target variable
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']


    ### Model Training and Evaluation ##
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)
    y_pred_ova = model_ova.predict(X_test)
    print("One-vs-All (OvA) Strategy:")
    print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


obesity_risk_pipeline(file_path, test_size=0.2)







