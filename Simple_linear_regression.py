import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

#Leer el data frame
df=pd.read_csv(url)

#varificar que lo ha leido
#print(df.sample(5))

#resumen estadístico de los datos
print(df.describe().T)

#elegir algunos datos
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#print(cdf.sample(9))

#Mostrar un histograma de ciertos datos
'''viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

#Graficas para comparar el gasto de combustible y las emisiones
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
#nos muestra que hay tres tipos de coches que siguen una relación lineal


#Otras graficas:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,10)
plt.show()
#tienen cierta correlación, pero menor.


#grafica plot entre clindrada y las emisiones
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()'''

#you will use engine size to predict CO2 emission with a linear regression model.
# You can begin the process by extracting the input feature and target output variables,
#  X and y, from the dataset.
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

'''Next, you will split the dataset into mutually exclusive training and testing sets. 
You will train a simple linear regression model on the training set and estimate its ability
to generalize to unseen data by using it to make predictions on the unseen testing data.
Since the outcome of each data point is part of the testing data, you have a means of
evaluating the out-of-sample accuracy of your model.
Now, you want to randomly split your data into train and test sets, using 80% of
the dataset for training and reserving the remaining 20% for testing. 
Which fraction to use here mostly depends on the size of your data, but typical training
sizes range from 20% to 30%. The smaller your data, the larger your training set needs to be
because it's easier to find spurious patterns in smaller data. The downside is that your 
evaluation of generalizability will have less reliability. Bigger is better when it
comes to data.'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn import linear_model

#crear el modelo
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data,
# with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)


# Coefficient and Intercept are the regression parameters determined by the model.
print ('Coefficients: ', regressor.coef_[0]) 
# with simple linear regression there is only one coefficient, here we extract it from
# the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

#Visualizar el modelo:
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Para medir el error:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))



#Ejercicio con el Consumo
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()

regr.fit(X_train.reshape(-1,1), y_train)


#Visualizar el modelo:
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regr.coef_ * X_train + regr.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

y_pred_test = regr.predict(X_test.reshape(-1,1))


print('Mean squared error: %2f' % mean_squared_error(y_test, y_pred_test))

