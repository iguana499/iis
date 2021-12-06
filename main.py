from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
import numpy as np


def rank_to_dict(ranks, names):
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


def sort_data(dict1):
    sorted_values = sorted(dict1.values())
    sorted_dict = {}

    for i in sorted_values:
        for k in dict1.keys():
            if dict1[k] == i:
                sorted_dict[k] = dict1[k]
                break
    return sorted_dict


np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] ** 5 + np.random.normal(0, 1))
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))

print(X)
print("")
print(Y)
lr = Lasso(alpha=0.1)
lr.fit(X, Y)

rfe = RFE(lr)
rfe.fit(X, Y)

f_regression = f_regression(X, Y)
f_regression = f_regression[0]


names = ["x%s" % i for i in range(1, 15)]
rank = [rank_to_dict(lr.coef_, names),
        rank_to_dict(f_regression, names),
        rank_to_dict(rfe.support_, names)]

for i in range(len(rank)):
    print(sort_data(rank[i]))