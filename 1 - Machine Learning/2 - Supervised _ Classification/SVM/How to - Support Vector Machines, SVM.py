"""
    Support Vector Machines

Based on: https://www.coursera.org/learn/machine-learning-with-python/
Created on Wed Apr 28 12:51:57 2021
@author: Dr.Eng. Xavier Fonseca
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


##############################################################################
##################### Getting dataset, preprocessing it as well  #############
##############################################################################
"""
    The ID field contains the patient identifiers. The characteristics of the 
    cell samples from each patient are contained in fields Clump to Mit. The 
    values are graded from 1 to 10, with 1 being the closest to benign.

    The Class field contains the diagnosis, as confirmed by separate medical 
    procedures, as to whether the samples are benign (value = 2) or malignant 
    (value = 4).
"""
cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()

# plotting the data nicelly. 
# Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size:
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

####################################
# Data pre-processing and selection
####################################
# Lets first look at columns data types:
cell_df.dtypes

# It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]


"""
    We want the model to predict the value of Class 
    (that is, benign (=2) or malignant (=4)). As this field can have one of 
    only two possible values, we need to change its measurement level to 
    reflect this.
"""
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]




##############################################################################
############################## Train/Test Split ##############################
##############################################################################
# Okay, we split our dataset into train and test set:
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


##############################################################################
#################### Creating and using predictive Model #####################
##############################################################################
"""
    The SVM algorithm offers a choice of kernel functions for performing its 
    processing. Basically, mapping data into a higher dimensional space is 
    called kernelling. The mathematical function used for the transformation is 
    known as the kernel function, and can be of different types, such as:

        1.Linear
        2.Polynomial
        3.Radial basis function (RBF)
        4.Sigmoid

    Each of these functions has its characteristics, its pros and cons, and its 
    equation, but as there's no easy way of knowing which function performs best 
    with any given dataset, we usually choose different functions in turn and 
    compare the results. Let's just use the default, 
    RBF (Radial Basis Function).
"""
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

# After being fitted, the model can then be used to predict new values:
yhat = clf.predict(X_test)
yhat [0:5]

#####################################################################
############################# Evaluating the Model ##################
# I am going to evaluate in 2 ways: Jaccard index, Confusion Matrix #
#####################################################################
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

"""
    Based on the count of each section, we can calculate precision and recall 
    of each label:

        Precision is a measure of the accuracy provided that a class label has 
        been predicted. It is defined by: precision = TP / (TP + FP)

        Recall is true positive rate. It is defined as: Recall =  TP / (TP + FN)

    So, we can calculate precision and recall of each class.

    F1 score: Now we are in the position to calculate the F1 scores for each 
    label based on the precision and recall of that label. The F1 score is the 
    harmonic average of the precision and recall, where an F1 score reaches its 
    best value at 1 (perfect precision and recall) and worst at 0. It is a good 
    way to show that a classifer has a good value for both recall and precision.
"""
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 






"""
    Lets try jaccard index for accuracy:
"""
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=2)






