import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data frame
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# to look at the data
print(df['custcat'].value_counts())

#mostrar la correlación en una matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()

# Para verlo más claro:
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)


### Prepare Data
X = df.drop('custcat', axis=1)
y = df['custcat']

# Normalizar X
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

### Training and predict the model
k = 4
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

yhat = knn_model.predict(X_test)

## Evaluation
print("Test set Accuracy: ", accuracy_score(y_test, yhat))


### Choose the correct value of k
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 



### EXERCISES:
# Q2: Run the training model for 30 values of k and then again for 100 values of k. 
Ks = 100
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 


'''
La baja precisión del modelo puede deberse a múltiples razones:

El modelo KNN depende completamente del espacio de características original en el momento de la predicción. 
Si las características no proporcionan límites claros entre las clases, el modelo KNN no puede compensarlo 
mediante optimización o transformación de características.

Cuando hay muchas características débilmente correlacionadas, el número de dimensiones aumenta. 
En espacios de alta dimensión, las distancias entre los puntos tienden a volverse más uniformes, 
lo que reduce el poder discriminativo del KNN.

El algoritmo trata todas las características por igual al calcular las distancias. 
Por eso, las características débilmente correlacionadas pueden introducir ruido o variaciones 
irrelevantes en el espacio de características, dificultando que KNN encuentre vecinos significativos.
'''









