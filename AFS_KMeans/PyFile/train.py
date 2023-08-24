import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from model import Kmeans

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = Kmeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()