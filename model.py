#!/usr/bin/env python3
import pandas as pd 
import os
import numpy as np
import math
import time
import collections
import concurrent.futures

class linear_regression(object):

    def __init__(self):
        pass

    def fit(self, X, ys, epochs, eta=0.01):
        X = np.asmatrix(X)
        ys = np.asmatrix(ys)

        num, n = ys.shape
        num, m = X.shape

        w_curr = np.asmatrix(np.zeros((m, n)))
        b_curr= np.asmatrix(np.zeros((1, n))) 
        y_curr = np.asmatrix(np.zeros((num, n)))
        y_diff = np.asmatrix(np.zeros((num, n)))

        for i in range(epochs):
            y_curr = np.matmul(X, w_curr) + b_curr
            y_diff = ys - y_curr
            w_error = -(2/num) * (np.matmul(X.T, y_diff))
            b_error = -(2/num) * y_diff.sum(axis=0)
            w_curr -= eta * w_error
            b_curr -= eta * b_error

        self.w = w_curr
        self.b = b_curr

    def predict(self, X):
        return np.matmul(X, self.w) + self.b


class logistic_regression(object):

    def __init__(self):
        pass

    def fit(self, X, ys, epochs, eta=0.01):
        X = np.asmatrix(X)
        ys = np.asmatrix(ys)

        num, n = ys.shape
        num, m = X.shape
        
        w_curr = np.asmatrix(np.zeros((m, n)))
        y_curr = np.asmatrix(np.zeros((num, n)))
        y_diff = np.asmatrix(np.zeros((num, n)))

        for i in range(epochs):
            y_curr = self.sigmoid(X, w_curr)
            y_diff = ys - y_curr
            w_error = np.matmul(X.T, -y_diff) / num 
            w_curr -= eta * w_error

        self.w = w_curr

    def predict(self, X):
        return self.sigmoid(X, self.w)

    def sigmoid(self, X, w):
        z = np.matmul(X, w)
        z = np.tanh(0.5 * z)
        return 0.5 * (z + 1)
