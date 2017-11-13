# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #all lines, columns -1
Y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy Variable Trap
X = X[:, 1:]


#create test and training data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

'''#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test Set Results
y_pred = regressor.predict(X_test)

#Preparing the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()

#removed 2 as that has confidence way higher than 5%
X_opt = X[:,[0,1,3,4,5]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()

#removed 1 as that has confidence way higher than 5%
X_opt = X[:,[0,3,4,5]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()

#removed 4 as that has confidence way higher than 5%
X_opt = X[:,[0,3,5]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()

#removed 5 as that has confidence way higher than 5%
X_opt = X[:,[0,3]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()