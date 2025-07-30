import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output

import warnings
warnings.filterwarnings('ignore')

# El objetivo es predecir si un cliente se dará de baja (churn) o no, basado en varias características.

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)



# We will us only 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

print(churn_df.head().T)

#Guardaremos en y la variable objetivo y en X las variables predictoras
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# Estandarizar las variables para que no destaquen unos datos respecto a otros
X_norm = StandardScaler().fit(X).transform(X)


# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

## Let's build the model using LogisticRegression from the Scikit-learn package and fit our model with train data set.
LR = LogisticRegression().fit(X_train, y_train)

# Predicting the test set results
yhat = LR.predict(X_test)

# Predicting the probabilities. We can get the probability of each class.
yhat_prob = LR.predict_proba(X_test)

'''La primera columna es la probabilidad de que el registro
pertenezca a la clase 0, y la segunda columna indica la probabilidad
de que pertenezca a la clase 1. Ten en cuenta que el sistema de 
predicción de clases utiliza un umbral de 0.5 para hacer la predicción. 
Esto significa que se predice la clase que tenga mayor probabilidad.'''

coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Evaluar el modelo
print("Log Loss: ", log_loss(y_test, yhat_prob))

'''Un valor positivo grande del coeficiente de regresión logística (LR Coefficient) para un determinado
campo indica que un aumento en ese parámetro conducirá a una mayor probabilidad de un resultado positivo,
es decir, clase 1. Un valor negativo grande indica lo contrario: que un aumento en ese parámetro reducirá
la probabilidad de que se obtenga una clase positiva. Un valor absoluto pequeño indica que ese campo 
tiene un efecto más débil sobre la clase predicha. Examinemos esto con los siguientes ejercicios.'''

## EJERCICIOS PRACTICOS

# a. Let us assume we add the feature 'callcard' to the original set of input features. 
# What will the value of log loss be in this case?

churn_df_a = pd.read_csv(url)
churn_df_a = churn_df_a[['callcard', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df_a['churn'] = churn_df_a['churn'].astype('int')
churn_df_a['callcard'] = churn_df_a['callcard'].astype('int')


X_a = np.asarray(churn_df_a[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard']])
y_a = np.asarray(churn_df_a['churn'])

X_a_norm = StandardScaler().fit(X_a).transform(X_a)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a_norm, y_a, test_size=0.2, random_state=42)
LR_a = LogisticRegression().fit(X_train_a, y_train_a)
yhat_a_prob = LR_a.predict_proba(X_test_a)
print("Log Loss with 'callcard': ", log_loss(y_test_a, yhat_a_prob))


# d. What happens to the log loss if we remove the feature 'equip' from
# the original set of input features?

churn_df_d = churn_df.drop(columns=['equip'])

X_d = np.asarray(churn_df_d[['tenure', 'age', 'address', 'income', 'ed', 'employ']])
y_d = np.asarray(churn_df_d['churn'])
X_d_norm = StandardScaler().fit(X_d).transform(X_d)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d_norm, y_d, test_size=0.2, random_state=42)
LR_d = LogisticRegression().fit(X_train_d, y_train_d)
yhat_d_prob = LR_d.predict_proba(X_test_d)
print("Log Loss without 'equip': ", log_loss(y_test_d, yhat_d_prob))


