import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=19)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)
log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train, y_train)
predict = log_reg.predict(X_test)
# X_test = np.column_stack((X_test, predict))
class0 = X_test[(predict[:] == 0)]
class1 = X_test[(predict[:] == 1)]
print(class0)

# plot
plt.title("Classification with Logistic Regression")
plt.scatter(class0[:, 0], class0[:, 1], color = 'red')
plt.scatter(class1[:, 0], class1[:, 1], color = 'green')
plt.show()

print("Number of wrong predictions is: ", end='')
print((predict != y_test).sum())