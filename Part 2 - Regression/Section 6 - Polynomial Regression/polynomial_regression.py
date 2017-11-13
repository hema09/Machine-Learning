# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #all lines, columns -1
Y = dataset.iloc[:, 2].values

#create test and training data
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

'''#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualizing the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear Regression results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff (Polynomial results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting the new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial Regression model
lin_reg_2.predict(poly_reg.fit_transform(6.5))