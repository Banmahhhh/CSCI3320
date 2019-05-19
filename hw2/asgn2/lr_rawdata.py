import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
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
df = df.dropna(axis=0)

# separate the data
length = df.index.size
train = int(length * 0.8)
test = length - train

X_train = df.iloc[0:train]
X_test = df.iloc[train:length]

# standardize the training and testing data of horsepower and price
# standardize, (x-mean)/sd,  mean = 0, var = 1
X_scaler = StandardScaler()

# fit and transform
# transform is based on fit_transform
X_train_scaled = pd.DataFrame(index = range(1, train+1), columns=['horsepower', 'price'])
X_test_scaled = pd.DataFrame(index = range(1, test+1), columns=['horsepower', 'price'])
X_train_scaled[['horsepower', 'price']] = X_scaler.fit_transform(X_train[['horsepower', 'price']])
X_test_scaled[['horsepower', 'price']] = X_scaler.transform(X_test[['horsepower', 'price']])
print(X_train_scaled)

# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train_scaled[['horsepower']], X_train_scaled[['price']])
print(regr.coef_)
pred = regr.predict(X_test_scaled[['horsepower']])

# # plot data
plt.title("Linear regression on cleaned and standardized test date")
plt.scatter(X_test_scaled['horsepower'], X_test_scaled['price'])
plt.plot(X_test_scaled['horsepower'], pred)
plt.show()
