import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


data = loadmat('C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\classification\\classification with multivar\\ex3data1.mat')


##################################     sickness data     ##################################


# print(data)
# print('===================================================')
# print(data['X'])
# print(data['y'])
# print('===================================================')
# print('X Shape = ' , data['X'].shape)
# print('Y Shape = ', data['y'].shape)
# print('===================================================')
# print (data['X'][0])   #all colmns=400
# print('===================================================')
# print(data['X'][0][155])
# print('===================================================')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def gradient_with_loop(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    return grad

def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X) 
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    return np.array(grad).ravel()

from scipy.optimize import minimize
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0] #5000
    params = X.shape[1] #400
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    print('all_theta shape ' , all_theta.shape)
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    print('X shape ' , X.shape)
    # labels are 1-indexed instead of 0-indexed
    #الامر الى بيعمل الclassification
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC',
        jac=gradient)
        all_theta[i-1,:] = fmin.x
    return all_theta


rows = data['X'].shape[0]      #5000
params = data['X'].shape[1]    #400


all_theta = np.zeros((10, params + 1))   #rows=10    اكنى عندى 10 انواع من المرض ب10 سيتات
print('all_theta \n' , all_theta)
print('all_theta shape \n' , all_theta.shape)   #10*401
print('===================================================')
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
print(X)
print('X Shape = ' , X.shape)
print('===================================================')
theta = np.zeros(params + 1)
print('theta \n' , theta )
print('===================================================')
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
print('y_0') 
print(y_0.shape)
print(y_0)
print('===================================================')
y_0 = np.reshape(y_0, (rows, 1))
print('y_0')
print(y_0.shape)
print(y_0)
print('===================================================')
print()
print('X.shape = ',X.shape)
print()
print('y.shape = ',y_0.shape)
print()
print('theta.shape = ',theta.shape)
print()
print('all_theta.shape = ',all_theta.shape)
print()
print('data array = ' , np.unique(data['y']))
print()