import numpy as np
from sklearn.datasets import make_classification
from models import Models


data = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)

X = np.array(data[0])
y = np.array(data[1])

X = X[:, np.newaxis, 1]

manager = Models(X, y)

manager.linear()
manager.perceptron()
manager.ridge()