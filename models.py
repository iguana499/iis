import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Perceptron


class Models:

    def __init__(self, X, y):
        self.X_train = X[:-20]
        self.X_test = X[-20:]
        self.y_train = y[:-20]
        self.y_test = y[-20:]

    def linear(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        print('Линейная регрессия')
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred))
        print('')

        plt.scatter(self.X_test, self.y_test, color='black')
        plt.plot(self.X_test, y_pred, color='blue', linewidth=3)
        plt.grid("on")
        plt.show()

    def ridge(self):
        clf = Ridge(alpha=1.0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        print('Гребневая полиномиальная регрессия')
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred))
        print('')

        plt.scatter(self.X_test, self.y_test, color='black')
        plt.plot(self.X_test, y_pred, color='blue', linewidth=3)
        plt.grid("on")
        plt.show()

    def perceptron(self):
        clf = Perceptron(tol=1e-3, random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        print('Персептрон')
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred))
        print('')

        plt.scatter(self.X_test, self.y_test, color='black')
        plt.plot(self.X_test, y_pred, color='blue', linewidth=3)
        plt.grid("on")
        plt.show()