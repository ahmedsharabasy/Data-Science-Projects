import imp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



#Read in the Ecommerce Customers csv file as a DataFrame called customers
print(os.listdir("C:\\Users\\20100\\Desktop\\data science\\machine learning"))
customers = pd.read_csv('C:\\Users\\20100\\Desktop\\data science\\machine learning\\projects\\regression\\Ecommerce Customers.csv')


#Check the head of customers, and check out its info() and describe() methods
print(customers.head())
print(customers.describe())
print(customers.info())


#Exploratory Data Analysis
#Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = customers)
plt.show()


####note
# JointGrid
# Set up a figure with joint and marginal views on bivariate data.

# PairGrid
# Set up a figure with joint and marginal views on multiple variables.

# jointplot
# Draw multiple bivariate plots with univariate marginal distributions.


#Do the same but with the Time on App column instead.
sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = customers)
plt.show()


#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on App',y ='Length of Membership', data = customers, kind='hex')
plt.show()


#Let's explore these types of relationships across the entire data set
sns.pairplot(customers,height=2)
plt.show()


#السؤال ده هو الغرض من الجراف سالت نفسى ايه اكتر شكل فيه ترابط من الانفاق السنوى
#Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
print("most correlated feature with Yearly Amount Spent? \nLength of Membership")


#Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.set(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)
plt.show()

#Training and Testing Data
#Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y= customers['Yearly Amount Spent']
#Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()       #Create an instance of a LinearRegression() model named lm.
lm.fit(X_train, y_train )
print(lm.coef_)

#Predicting Test Data
predictions = lm.predict(X_test)      #Use lm.predict() to predict off the X_test set of the data.
plt.scatter(y_test, predictions)  #Create a scatterplot of the real test values versus the predicted values.
plt.ylabel('Predicted')
plt.xlabel('Y test')
plt.show()

#Evaluating the Model
#Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas
import sklearn.metrics as metrics
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))


#Residuals
#You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data.
#Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot(y_test-predictions)
plt.show()

#Recreate the dataframe below.
conclosion=pd.DataFrame(lm.coef_ , X.columns, columns=['Coeffecient'])
print(conclosion)

# Do you think the company should focus more on their mobile app or on their website???
# answer.  The company should focus on the mobile app

