#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Importing all the packages to implement simple Linear Regression with scikit-learn
        -    Use scikit-learn to implement simple Linear Regression
        -    Create a model, train,test and use the model
        


    Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) 
    to minimize the 'residual sum of squares' between the actual value y in the 
    dataset, and the predicted value yhat using linear approximation.


Created on Wed Apr 21 11:25:21 2021
@author: Dr.Eng. Xavier Fonseca
https://www.coursera.org/learn/machine-learning-with-python/ungradedLti/X7V0r/lab-simple-linear-regression
"""

#importing necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np




# Reading data, plotting data set

# Read the data you already downloaded from online.
# !wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

########################################################
############### Reading the data in ####################
########################################################

df = pd.read_csv("/Users/Xavier/GoogleHDD/Trabalho/1 - Cursos/Machine Learning with Python - Coursera/FuelConsumption.csv")

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
    As mentioned before, Coefficient and Intercept in the simple linear 
    regression, are the parameters of the fit line. Given that it is a simple 
    linear regression, with only 2 parameters, and knowing that the parameters 
    are the intercept and slope of the line, 
    
    sklearn can estimate them directly from our data. 
    
    Notice that all of the data must be available to traverse and calculate the 
    parameters.

"""

# Using sklearn package to model/train data.
# numpy.asanyarray() converts list inputs into array

from sklearn import linear_model
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regression.coef_)
print ('Intercept: ',regression.intercept_)

# We can plot the fit line over the data:

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")




########################################################
##################### Evaluation #######################
########################################################
"""
    We compare the actual values and predicted values to calculate the accuracy 
    of a regression model. Evaluation metrics provide a key role in the 
    development of a model, as it provides insight to areas that require 
    improvement.
    
    There are different model evaluation metrics, lets use MSE here  
    to calculate the accuracy of our model based on the test set.
    
    - Mean absolute error: It is the mean of the absolute value of the errors. 
        This is the easiest of the metrics to understand since it’s just 
        average error.
    
    - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the 
        squared error. It’s more popular than Mean absolute error because the 
        focus is geared more towards large errors. This is due to the squared 
        term exponentially increasing larger errors in comparison to smaller 
        ones.
        
    - Root Mean Squared Error (RMSE).
    
    - R-squared is not error, but is a popular metric for accuracy of your 
    model. It represents how close the data are to the fitted regression line. 
    The higher the R-squared, the better the model fits your data. Best 
    possible score is 1.0 and it can be negative 
    (because the model can be arbitrarily worse).
"""

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regression.predict(test_x)

# o % e vez de virgula no print tem a ver com o %.2f, para que %.2f não seja 
# impresso na mensagem, mas de facto formate o resultado do parametro seguinte
print("Mean absolute error (MAE): %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )















