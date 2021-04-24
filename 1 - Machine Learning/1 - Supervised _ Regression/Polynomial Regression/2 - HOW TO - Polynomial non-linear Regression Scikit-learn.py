"""
Created on Thu Apr 22 11:06:12 2021

@author: Xavier Fonseca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Example of a simple equation, of degree 1, for example y = 2𝑥 + 3.
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


###############################################################################
########################### Example of Cubic function #########################
###############################################################################
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
"""
    As you can see, this function has 𝑥3 and 𝑥2 as independent variables. Also, 
    the graphic of this function is not a straight line over the 2D plane. So 
    this is a non-linear function.
"""
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


###############################################################################
##################### Example of Quadratic function X^2 #######################
###############################################################################
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

###############################################################################
##################### Example of Exponential function #########################
###############################################################################
# Y = a + b c^X
X = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

Y= np.exp(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

###############################################################################
##################### Example of Logarithmic function #########################
###############################################################################
# Y = log(x)
# Please consider that instead of $x$, we can use $X$, which can be polynomial 
# representation of the $x$'s. In general form it would be written as Y = log(X)

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

###############################################################################
##################### Example of Sigmoidal/Logistic function ##################
###############################################################################
# Y = a + b / (1 + c ^ (X - d))
X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()




##############################################################################
###################### Getting dataset and plotting it #######################
##############################################################################

df = pd.read_csv("/Users/Xavier/GoogleHDD/Trabalho/1 - Cursos/Machine Learning with Python - Coursera/1 - Supervised _ Polynomial Regressions/china_gdp.csv",sep=';')

# take a look at the dataset
df.head(10)


plt.figure(figsize=(8,5))
x_data, y_data = (df["YEAR"].values, df["GDP"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

"""
    From an initial look at the plot, we determine that the logistic function 
    could be a good approximation, since it has the property of starting with 
    a slow growth, increasing growth in the middle, and then decreasing again 
    at the end;
    
    𝛽1: Controls the curve's steepness,
    𝛽2: Slides the curve on the x-axis. 
"""

##############################################################################
###################### Building the model (finding coefficients) #############
##############################################################################

# Let's build our regression model and initialize its parameters. 
# this is the function sigmoid/logistic I'm going to use in my model a bit ahead
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

## Lets look at a sample sigmoid line that might fit with the data:
#beta_1 = 0.10
#beta_2 = 1990.0

##logistic function
#Y_pred = sigmoid(x_data, beta_1 , beta_2)

##plot initial prediction against datapoints
#plt.plot(x_data, Y_pred*15000000000000.)
#plt.plot(x_data, y_data, 'ro')



"""
How we find the best parameters for our fit line?

    Our task here is to find the best parameters for our model. 
    Lets first normalize our x and y.

    Then, we can use curve_fit which uses non-linear least squares to fit our sigmoid 
    function, to data. Optimal values for the parameters so that the sum of the 
    squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

    popt are our optimized parameters (all of them)
    𝛽1 (curve's steepness) and    
    𝛽2: (Slides the curve on the x-axis).

"""

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata) # creates our model
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Now we plot our resulting regression model.

x = np.linspace(1960, 2021, 61)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)   # predicts values with the correct parameters popt
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

y[60]*np.power(10,10)


##############################################################################
########################## Testing the accuracy of model #####################
##############################################################################

# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]


# build the model using train set
# popt are our optimized parameters 
# pcov are covariance of the parameters
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
# popt has 2 values in. Doing *popt passes these two objects instead of 1
# estimated_k, estimated_x0 = popt
y_hat = sigmoid(test_x, *popt) #this is my model; y_hat is my prediction


# evaluation (you can only evaluate if you save some data (20% for example) to verify)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )






