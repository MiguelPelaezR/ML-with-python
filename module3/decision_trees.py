###  medical researcher compiling data for a study ###

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')



# Read data
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

# Use LabelEncoder to converte categorical into numerica dataset
my_data['Sex'] = LabelEncoder().fit_transform(my_data['Sex'])
my_data['BP'] = LabelEncoder().fit_transform(my_data['Sex'])
my_data['Cholesterol'] = LabelEncoder().fit_transform(my_data['Sex'])
#print(my_data.head())

#print(my_data.isnull().sum()) #no hay datos nulos

# Evaluate  the correlation to the target variable
# Cambiar las drogas por valores
custom_map = {'drugA':0, 'drugB':1, 'drugC':2, 'drugX':3, 'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

my_data = my_data.drop(['Drug'], axis=1)
print(my_data.corr().T)
''' Correlación principalmente con Na_to_K y BP '''

# Watch the distribution of the drugs:
category_counts = my_data['Drug_num'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()


### MODELING  ###

y = my_data['Drug_num']
X = my_data.drop(['Drug_num'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
drugTree.fit(X_train, y_train)

# Evaluation
tree_pred = drugTree.predict(X_test)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, tree_pred))

# Visualize the tree:
plot_tree(drugTree)
plt.show()





