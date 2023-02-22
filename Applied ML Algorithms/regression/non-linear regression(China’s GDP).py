import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\regression\\china_gdp_1960.csv')
print(dataset.head(10))

plt.figure(figsize=(8,5))
x_data, y_data = (dataset["Year"].values, dataset["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


#Choosing a Model
# From an initial look at the plot, we determine that the logistic function could be a good approximation,
# since it has the property of starting with slow growth,
# increasing growth in the middle, and then decreasing again at the end as illustrated below:4


X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()




#######################################(((logistic function)))################################################
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))    #formula for the logistic function
    return y
beta_1 = 0.10
beta_2 = 1990.0
Y_pred = sigmoid(x_data, beta_1 , beta_2)
#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#HOW WE FIND THE BEST PARAMETERS FOR OUR FIT LINE?
#We can use curve_fit which uses non-linear least squares to fit our sigmoid function, to data.
#Optimal values for the parameters so that the sum of the squared residuals of sigmoid (xdata, *popt) – ydata is minimized.
#*popt are our optimized parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)    #Use non-linear least squares to fit a function, f, to data
# popt : array
#     Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
# pcov : 2-D array
#     The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
#     To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
print('the final parameters')
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#Now we plot our resulting regression model.
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


#Now, let’s find the accuracy of our model.
# split data into train/test
msk = np.random.rand(len(dataset)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
