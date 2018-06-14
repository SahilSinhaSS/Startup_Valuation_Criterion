#Multiple Linear Regression

#Importing libraraies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, -1:].values

#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
sc_X = LabelEncoder()
X[:, 3]= sc_X.fit_transform(X[:, 3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

#Splitting into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


#Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y,exog = x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                else:
                    continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Modeled, y_train)

#prediction
y_pred = regressor.predict(X_test)
