###  build a model that predicts if a credit card transaction is fraudulent or not. ###

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')


# Load data set
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
df = pd.read_csv(url)

labels = df.Class.unique() # Array de los valores en la columna class

sizes = df.Class.value_counts().values

#plot the class value. Grafica de tarta
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

# which features affect the model
correlation_values = df.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))
plt.show()

## Data processing
# Standarize the data
df.iloc[:,1:30] = StandardScaler().fit_transform(df.iloc[:,1:30])
data_matrix = df.values

X = data_matrix[:,1:30]
y = data_matrix[:,30]

X = normalize(X, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### BUILD THE MODEL ###
## THE DECISION TREE

# dar mas peso a las clases menos representadas
w_train = compute_sample_weight('balanced', y_train)

# Creamos arbol de decisión
dec_tree = DecisionTreeClassifier(max_depth=4, random_state=35)
dec_tree.fit(X_train, y_train, sample_weight=w_train)

## THE SUPPORT VECTOR MACHINE
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

### EVALUTE THE MODELS
# Tree:
y_pred_dec_tree = dec_tree.predict_proba(X_test)[:,1]

roc_auc_dec_tree = roc_auc_score(y_test, y_pred_dec_tree)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dec_tree))

# SVM
y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))


### EXERCISES
# Q1: Currently, we have used all 30 features of the dataset for training
#  the models. Use the corr() function to find the top 6 features of the dataset to train the models on.
correlation_values = abs(df.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
print(correlation_values)

# Q2. Using only these 6 features, modify the input variable for training.
X = data_matrix[:,[3,10,12,14,16,17]]

# Q3
X = normalize(X, norm='l1')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

w_train = compute_sample_weight('balanced', y_train)

# Creamos arbol de decisión
dec_tree = DecisionTreeClassifier(max_depth=4, random_state=35)
dec_tree.fit(X_train, y_train, sample_weight=w_train)
## THE SUPPORT VECTOR MACHINE
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)

### EVALUTE THE MODELS
# Tree:
y_pred_dec_tree = dec_tree.predict_proba(X_test)[:,1]
roc_auc_dec_tree = roc_auc_score(y_test, y_pred_dec_tree)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dec_tree))

# SVM
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

'''
With a larger set of features, SVM performed relatively better in comparison to the Decision Trees.
Decision Trees benefited from feature selection and performed better.
SVMs may require higher feature dimensionality to create an efficient decision hyperplane.
'''


