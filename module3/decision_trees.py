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
print(my_data.head())








