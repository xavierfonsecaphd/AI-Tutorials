"""
Created on Thu Apr 22 11:06:12 2021

@author: Xavier Fonseca
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

df = pd.read_csv("/Users/Xavier/GoogleHDD/Trabalho/1 - Cursos/Machine Learning with Python - Coursera/1 - Supervised _ Polynomial Regressions/1 - FuelConsumption Data.csv")

# take a look at the dataset
df.head()

# Lets select some features that we want to use for regression.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Lets plot Emission values with respect to Engine size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

##############################################################################
###################### Creating train and test dataset #######################
##############################################################################
"""
    Train/Test Split involves splitting the dataset into training and testing 
    sets respectively, which are mutually exclusive. After which, you train 
    with the training set and test with the testing set.
"""
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

"""
    Sometimes, the trend of data is not really linear, and looks curvy. In this 
    case we can use Polynomial regression methods. In fact, many different 
    regressions exist that can be used to fit whatever the dataset looks like, 
    such as quadratic, cubic, and so on, and it can go on and on to infinite 
    degrees.

    In essence, we can call all of these, polynomial regression, where the 
    relationship between the independent variable x and the dependent variable 
    y is modeled as an nth degree polynomial in x. Lets say you want to have a 
    polynomial regression (let's make 2 degree polynomial):

                           $$y = b + \theta_1  x + \theta_2 x^2$$

    Now, the question is: how we can fit our data on this equation while we 
    have only x values, such as Engine Size? 
    Well, we can create a few additional features: 1, $x$, and $x^2$.
"""
"""
    PolynomialFeatures() function in Scikit-learn library, drives a new 
    feature sets from the original feature set. That is, a matrix will be 
    generated consisting of all polynomial combinations of the features with 
    degree less than or equal to the specified degree. For example, lets say 
    the original feature set has only one feature, _ENGINESIZE_. Now, if we 
    select the degree of the polynomial to be 2, then it generates 3 features, 
    degree=0, degree=1 and degree=2: 
"""

##############################################################################
###################### Transforming data (features) into a polynomial ########
################################# set of features ############################
##############################################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

"""
    Now, we can deal with it as 'linear regression' problem. Therefore, this 
    polynomial regression is considered to be a special case of traditional 
    multiple linear regression. So, you can use the same mechanism as linear 
    regression to solve such a problems.

    so we can use LinearRegression() function to solve it:
"""
##############################################################################
###################### Training the model (finding coefficients) #############
##############################################################################
model = linear_model.LinearRegression()
train_y_ = model.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', model.coef_)
print ('Intercept: ',model.intercept_)

"""
    As mentioned before, Coefficient and Intercept , are the parameters of the 
    fit curvy line. Given that it is a typical multiple linear regression, with 
    3 parameters, and knowing that the parameters are the intercept and 
    coefficients of hyperplane, sklearn has estimated them from our new set of 
    feature sets. Lets plot it:
"""
##############################################################################
###################### Plotting training data with our Model #################
##############################################################################
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = model.intercept_[0]+ model.coef_[0][1]*XX+ model.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

##############################################################################
############################### Testing model ################################
##############################################################################
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = model.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )






