import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class Manager:

    def __init__(self, data):
        self.data = data
        self.keys = data.keys()

    # удаление шумов и преобразование данных
    def data_preparation(self):
        self.data[self.keys[5]].update(self.data[self.keys[5]].replace(np.nan, self.data[self.keys[5]].median()))

        self.data[self.keys[10]].update(self.data[self.keys[10]].replace(np.nan, "0"))
        for i in range(len(self.data[self.keys[10]])):
            if self.data[self.keys[10]][i] != "0":
                self.data[self.keys[10]].update(self.data[self.keys[10]].replace(self.data[self.keys[10]][i], "1"))

        self.data[self.keys[10]] = pd.to_numeric(self.data[self.keys[10]])

        self.data['Embarked'].fillna('S', inplace=True)
        self.data = pd.concat([self.data, pd.get_dummies(self.data['Embarked'], prefix="Embarked")], axis=1)
        self.data.drop(['Embarked'], axis=1, inplace=True)
        return self.data

    # тестирование
    def test_tree(self, X_train, y_train, X_test, y_test):
        rfc = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=21)
        rfc.fit(X_train, y_train)
        print(rfc.score(X_test, y_test))

    # обучение
    def learn_tree(self, X_train, y_train):
        clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        return clf
