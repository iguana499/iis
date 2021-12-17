import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from sklearn.cluster import DBSCAN

np.random.seed(42)


# Function for creating datapoints in the form of a circle
def PointsInCircum(r, n=100):
    return [(math.cos(2 * math.pi / n * x) * r + np.random.normal(-30, 30),
             math.sin(2 * math.pi / n * x) * r + np.random.normal(-30, 30)) for x in range(1, n + 1)]


# Creating data points in the form of a circle
df = pd.DataFrame(PointsInCircum(500, 1000))
df = df.append(PointsInCircum(300, 700))
df = df.append(PointsInCircum(100, 300))

# Adding noise to the dataset
df = df.append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])

plt.figure(figsize=(10, 10))
plt.scatter(df[0], df[1], s=15, color='grey')
plt.title('Dataset', fontsize=20)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()

dbscan_opt = DBSCAN(eps=30, min_samples=6)
dbscan_opt.fit(df[[0, 1]])

df['DBSCAN_opt_labels'] = dbscan_opt.labels_
df['DBSCAN_opt_labels'].value_counts()

dbscan=DBSCAN()
dbscan.fit(df[[0,1]])

plt.figure(figsize=(10, 10))
plt.scatter(df[0], df[1], c=df['DBSCAN_opt_labels'], cmap=matplotlib.colors.ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"]), s=15)
plt.title('DBSCAN Clustering', fontsize=20)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()
