import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\classification\\exam grades.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print('data = ')
print(data.head(10) )
print()
print('data.describe = ')
print(data.describe())
print(data.info())

accepted = data[data['Admitted'].isin([1])]
rejected = data[data['Admitted'].isin([0])]
print('admitted student',accepted)
print()
print('nonadmitted student',rejected)


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(accepted['Exam 1'], accepted['Exam 2'], s=50, c='b', marker='o',label='Admitted')
ax.scatter(rejected['Exam 1'], rejected['Exam 2'], s=50, c='r', marker='x', label='NotAdmitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.plot()

#calc the cost
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# nums = np.arange(-10, 10, step=1)
# fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(nums, sigmoid(nums), 'r')

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]    #100*4   #=4
X = data.iloc[:,0:cols-1]        # iloc >>> بتخش بااندكس معين وتبدأ بتحصيل البيانات
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
print()
print('X.shape = ' , X.shape)
print('theta.shape = ' , theta.shape)
print('y.shape = ' , y.shape)
print(theta)

thiscost = cost(theta, X, y)
print()
print('cost = ' , thiscost)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

costafteroptimize = cost(result[0], X, y)
print()
print('cost after optimize = ' , costafteroptimize)
print()
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(correct) % len(correct))
print ('accuracy = {0}%'.format(accuracy))