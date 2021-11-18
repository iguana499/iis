import pandas
import eli5
from tree import Manager

data = pandas.read_csv("titanic.csv")

PASSENGER_ID = "PassengerId"
SURVIVED = "Survived"
PASSENGER_CLASS = "Pclass"
NAME = "Name"
SEX = "Sex"
AGE = "Age"
SIBSP = "SibSp"
PARCH = "Parch"
TICKET = "Ticket"
FARE = "Fare"
CABIN = "Cabin"
EMBARKED = "Embarked"




def work(data):
    manager = Manager(data)

    data = manager.data_preparation()

    # отделение нужных данных
    X = data.drop([PASSENGER_ID, SURVIVED, PASSENGER_CLASS, NAME, SEX, SIBSP, PARCH, TICKET, FARE], axis=1)
    y = data[SURVIVED]
    print(X.head())
    # выборка данных для обучения и тестирования
    X_train = X[:-200]
    X_test = X[-200:]
    y_train = y[:-200]
    y_test = y[-200:]

    clf = manager.learn_tree(X_train, y_train)
    manager.test_tree(X_train, y_train, X_test, y_test)
    return [clf, X_train]


result = work(data)
# отпределение веса
print(eli5.explain_weights_sklearn(result[0], feature_names=result[1].columns.values))
