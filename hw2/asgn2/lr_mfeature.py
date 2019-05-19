import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
import seaborn
seaborn.set()

# read the data
df = pd.read_csv('imports-85.data',
    header = None,
    names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
        'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
        'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'], 
        na_values=('?'))

# drop instances with NaN
df = df.dropna()

# separate the data
length = df.index.size

# standardize the training and testing data of horsepower and price
# standardize, (x-mean)/sd  mean = 0, var = 1
X_scaler = StandardScaler()

# fit and transform
# transform is based on fit_transform
X = X_scaler.fit_transform(df[['city-mpg', 'horsepower', 'engine-size', 'peak-rpm']])
Y = X_scaler.fit_transform(df[['price']])

X_prime = np.insert(X, 0, 1, axis = 1).astype(float)

# multiple linear regression
temp = np.dot(np.transpose(X_prime), Y)
temp2 = np.dot(np.transpose(X_prime), X_prime)
theta = np.dot(np.linalg.inv(temp2), temp)
print("Parameter theta calculated by normal equation: ", end = '')
print(theta)

# SGD regression
clf = linear_model.SGDRegressor(loss = 'squared_loss', max_iter=1000)
clf.fit(X, Y)
print("Parameter theta calculated by SGD: " + str(clf.intercept_) + " " + str(clf.coef_))