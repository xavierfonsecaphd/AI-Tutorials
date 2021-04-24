#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Importing all the packages to implement Multiple Linear Regression with scikit-learn
        -    Use scikit-learn to implement Multiple Linear Regression
        -    Create a model, train,test and use the model
        


    Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) 
    to minimize the 'residual sum of squares' between the actual value y in the 
    dataset, and the predicted value yhat using linear approximation.


Created on Wed Apr 21, 2021
@author: Dr.Eng. Xavier Fonseca
https://www.coursera.org/learn/machine-learning-with-python/ungradedLti/X7V0r/lab-simple-linear-regression
"""

#importing necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline



# Reading data, plotting data set

# Read the data you already downloaded from online.
# !wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

########################################################
############### Reading the data in ####################
########################################################

df = pd.read_csv("/Users/Xavier/GoogleHDD/Trabalho/1 - Cursos/Machine Learning with Python - Coursera/1 - FuelConsumption Data.csv")

# take a look at the dataset
df.head()

########################################################
############### Data Exploration #######################
########################################################

# summarize the data
df.describe()

# Lets select some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# We can plot each of these featues:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Now, lets plot each of these features vs the Emission, to see how linear is their relation:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# plotting cylinders to see their emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()



"""
Creating train and test dataset

    Train/Test Split
    involves splitting the dataset into training and testing sets respectively, 
    which are mutually exclusive. After which, you train with the training set 
    and test with the testing set. This will provide a more accurate evaluation 
    on out-of-sample accuracy because the testing dataset is not part of the 
    dataset that have been used to train the data. It is more realistic for 
    real world problems.
    
    This means that we know the outcome of each data point in this dataset, 
    making it great to test with! And since this data has not been used to 
    train the model, the model has no knowledge of the outcome of these data 
    points. So, in essence, it is truly an out-of-sample testing.
    
    Lets split our dataset into train and test sets, 80% of the entire data for 
    training, and the 20% for testing. We create a mask to select random rows 
    using np.random.rand() function
"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#### Train data distribution | if you want to see it plotted | optional
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

########################################################
######### Modeling Simple Regression Model #############
########################################################

"""
    In reality, there are multiple variables that predict the Co2emission. When 
    more than one independent variable is present, the process is called 
    multiple linear regression. For example, predicting co2emission using 
    FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars. The good thing here 
    is that Multiple linear regression is the extension of simple linear 
    regression model.
    
    As mentioned before, Coefficient and Intercept , are the parameters of the 
    fit line. Given that it is a multiple linear regression, with 3 parameters, 
    and knowing that the parameters are the intercept and coefficients of 
    hyperplane, sklearn can estimate them from our data. Scikit-learn uses 
    plain Ordinary Least Squares method to solve this problem.

"""

# Using sklearn package to model/train data.
# numpy.asanyarray() converts list inputs into array

from sklearn import linear_model
regression = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit (x, y)
# The coefficients
print ('Coefficients: ', regression.coef_)




########################################################
############ Ordinary Least Squares (OLS) ##############
########################################################
"""
Ordinary Least Squares (OLS)

    OLS is a method for estimating the unknown parameters in a linear 
    regression model. OLS chooses the parameters of a linear function of a set 
    of explanatory variables by minimizing the sum of the squares of the 
    differences between the target dependent variable and those predicted by 
    the linear function. In other words, it tries to minimizes the sum of 
    squared errors (SSE) or mean squared error (MSE) between the target 
    variable (y) and our predicted output (𝑦̂) over all samples in the dataset.

    OLS can find the best parameters using of the following methods:

        - Solving the model parameters analytically using closed-form equations

        - Using an optimization algorithm 
        (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
"""

########################################################
############ Making Prediction Model with OLS ##########
########################################################

# prediction model y_hat
y_hat= regression.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"  % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regression.score(x, y))











