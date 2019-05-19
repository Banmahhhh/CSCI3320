import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# prediction and score
clf = LinearRegression()
clf.fit(X_train_poly, y_train)
print(clf.coef_)
print(clf.intercept_)
print("Linear regression (order 5) score is: ", end='')
print(clf.score(X_test_poly, y_test))

# predict on xx
xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = clf.predict(xx_poly)

# plot xx and yy_ploy 
plt.figure(1)
plt.title("Linear regression (order 5) result")
plt.plot(xx, yy_poly)
plt.scatter(X_test, y_test)

# ridge regression
ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
print(ridge_model.coef_)
print(ridge_model.intercept_)

# predict on xx
yy_ridge = ridge_model.predict(xx_poly)
print("Ridge regression (order 5) score is: ", end='')
print(ridge_model.score(X_test_poly, y_test))

# plot xx and yy_ridge
plt.figure(2)
plt.title("Ridge regression (order 5) result")
plt.plot(xx, yy_ridge)
plt.scatter(X_test, y_test)
# plt.show()