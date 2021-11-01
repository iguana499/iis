import pandas
import eli5
from tree import Manager


data = pandas.read_csv("titanic.csv")


''' --Keys--
0 - PassengerId
1 - Survived
2 - Pclass
3 - Name
4 - Sex
5 - Age
6 - SibSp
7 - Ticket
8 - Fare
9 - Cabin
10 - Embarked
'''
keys = data.keys()


manager = Manager(data)

data = manager.data_preparation()

# отделение нужных данных
X = data.drop([keys[0], keys[1], keys[2], keys[3], keys[4], keys[6], keys[7], keys[8], keys[9]], axis=1)
y = data[keys[1]]
print(X.head())
# выборка данных для обучения и тестирования
X_train = X[:-200]
X_test = X[-200:]
y_train = y[:-200]
y_test = y[-200:]

clf = manager.learn_tree(X_train,y_train)
manager.test_tree(X_train,y_train,X_test,y_test)

# отпределение веса
print(eli5.explain_weights_sklearn(clf, feature_names=X_train.columns.values))