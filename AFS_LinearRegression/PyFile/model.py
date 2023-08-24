# import some importation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# building class Linear Regression
class LinearRegression:
    # Assign some variable
    # lr : learning rate
    # n_iters : number of training epochs
    def __init__(self, lr = 0.001, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None # Assign weights
        self.bias = None # bias you can know as w0

    # fit function
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # Weights start with 0
        self.bias = 0 # Start bias in 0

        for _ in range(self.n_iters):
            # Get y predict
            y_pred = np.dot(X, self.weights) + self.bias

            # Get deviation of weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update new weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        # Prediction y with new input to test
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
