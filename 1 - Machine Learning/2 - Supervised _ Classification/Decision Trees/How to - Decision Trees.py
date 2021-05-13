"""
Created on Tue Apr 27, 2021 

    Decision Trees
    
    You will learn a popular machine learning algorithm, Decision Tree. You 
    will use this classification algorithm to build a model from historical 
    data of patients, and their response to different medications. Then you use 
    the trained decision tree to predict the class of a unknown patient, or to 
    find a proper drug for a new patient.
    
    Imagine that you are a medical researcher compiling data for a study. You 
    have collected data about a set of patients, all of whom suffered from the 
    same illness. During their course of treatment, each patient responded to 
    one of 5 medications, Drug A, Drug B, Drug c, Drug x and y.

    Part of your job is to build a model to find out which drug might be 
    appropriate for a future patient with the same illness. The feature sets of 
    this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and 
    the target is the drug that each patient responded to.

    It is a sample of multiclass classifier, and you can use the training part 
    of the dataset to build a decision tree, and then use it to predict the 
    class of a unknown patient, or to prescribe it to a new patient. 

@author: Dr. Eng. Xavier Fonseca
@Link:   Based on: https://www.coursera.org/learn/machine-learning-with-python/ungradedLti/SzC4i/lab-decision-trees
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


##############################################################################
##################### Getting dataset, and converting  #######################
################## categorical data into numerical values ####################
##############################################################################
# and using pandas for dataframe structures
my_data = pd.read_csv("drug200.csv",sep=',')


my_data[0:5]
len(my_data)    # number of records in this dataset (my_data.shape())

"""
    X as the Feature Matrix (data of my_data)
    y as the response vector (target)

"""
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5] # looking at the first 5 rows

"""
    As you may figure out, some features in this dataset are categorical such 
    as Sex or BP. Unfortunately, Sklearn Decision Trees do not handle 
    categorical variables. But still we can convert these features to numerical 
    values. pandas.get_dummies() Convert categorical variable into dummy/indicator 
    variables
"""
from sklearn import preprocessing
labeleEncoder_sex = preprocessing.LabelEncoder()
labeleEncoder_sex.fit(['F','M'])
X[:,1] = labeleEncoder_sex.transform(X[:,1]) # with X[:,1], you're selecting which column of our data to apply data re-labelling


labeleEncoder_BP = preprocessing.LabelEncoder()
labeleEncoder_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = labeleEncoder_BP.transform(X[:,2])


labeleEncoder_Chol = preprocessing.LabelEncoder()
labeleEncoder_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = labeleEncoder_Chol.transform(X[:,3]) 

X[0:5]

# Now we can fill the target variable.
y = my_data["Drug"]
y[0:5]


##############################################################################
############################## Train/Test Split ##############################
##############################################################################
""" We will be using train/test split on our decision tree. Let's import 
    train_test_split from sklearn.cross_validation.
    
    Now train_test_split will return 4 different parameters. 
    We will name them:
        X_trainset, X_testset, y_trainset, y_testset 
    
    The train_test_split will need the parameters:
        X, y, test_size=0.3, and random_state=3. (test size is 30%, and random-state is to replicate the division)
    
    The X and y are the arrays required before the split, the test_size 
    represents the ratio of the testing dataset, and the random_state ensures 
    that we obtain the same splits.
"""
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# just printing the shape of the vectors X_trainset and y_trainset for fun
#print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
#print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))



##############################################################################
#################### Creating and using predictive Model #####################
##############################################################################
"""
    We will first create an instance of the DecisionTreeClassifier called 
    drugTree. Inside of the classifier, specify criterion="entropy" so we can 
    see the information gain of each node. 
    
    max_depth is specified. If not, the maximum depth of the tree will be untill
    all leaves are pure or until leaves contain less than a certain number of samples.
"""
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# Next, we will fit the data with the training feature matrix X_trainset and 
# training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)


# Let's make some predictions on the testing dataset and store it into a 
# variable called predTree. 
predictionTree = drugTree.predict(X_testset)

# You can print out predTree and y_testset if you want to visually compare the prediction to the actual values.
print (predictionTree [0:5])
print (y_testset [0:5])

##############################################################################
############################# Plotting Decision Tree #########################
##############################################################################

# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
#!conda install -c conda-forge pydotplus -y
#!conda install -c conda-forge python-graphviz -y
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


##############################################################################
############################# Evaluating the Model ###########################
##############################################################################
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predictionTree))



