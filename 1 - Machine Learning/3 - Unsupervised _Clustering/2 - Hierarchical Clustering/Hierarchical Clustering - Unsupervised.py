"""
    Hierarchical Clustering - Agglomerative
    
    We will be looking at a clustering technique, which is Agglomerative 
    Hierarchical Clustering. Remember that agglomerative is the bottom up 
    approach.

    In this code we will be looking at Agglomerative clustering, which is more 
    popular than Divisive clustering. 
    We will also be using Complete Linkage as the Linkage Criteria.
    
    NOTE: You can also try using Average Linkage wherever Complete Linkage 
    would be used to see the difference! 
    

Created on Fri Apr 30 14:44:34 2021
@author: Dr. Eng. Xavier Fonseca
Based on: https://www.coursera.org/learn/machine-learning-with-python
"""
import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs

##############################################################################
####       Hierarchical Clustering on a Randomly Generated Dataset        ####
##############################################################################
"""
    We will be generating a set of data using the make_blobs class.

    Input these parameters into make_blobs:

        n_samples: The total number of points equally divided among clusters.
            Choose a number from 10-1500
            
        centers: The number of centers to generate, or the fixed center locations.
            Choose arrays of x,y coordinates for generating the centers. 
            Have 1-10 centers (ex. centers=[[1,1], [2,5]])
            
        cluster_std: The standard deviation of the clusters. The larger the 
            number, the further apart the clusters
            Choose a number between 0.5-1.5
            
    Save the result to X1 and y1
"""
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

# Plot the scatter plot of the randomly generated data
plt.scatter(X1[:, 0], X1[:, 1], marker='o') 



##############################################################################
#############                Building the Model                    ###########
##############################################################################
"""
Agglomerative Clustering

    We will start by clustering the random data points we just created.

    The Agglomerative Clustering class will require two inputs:

        n_clusters: The number of clusters to form as well as the number of 
        centroids to generate. Value will be: 4
            
        linkage: Which linkage criterion to use. The linkage criterion determines 
            which distance to use between sets of observation. The algorithm 
            will merge the pairs of cluster that minimize this criterion. Value
            will be: 'complete'. Note: It is recommended you try everything 
            with 'average' as well.
"""
#Save the result to a variable called agglom Model
agglomModel = AgglomerativeClustering(n_clusters = 4, linkage = 'average')

# Fit the model with X1 and y1 from the generated data above.
agglomModel.fit(X1,y1)

 
##############################################################################
#############                Visualizing the Model                 ###########
#############    Run the following code to show the clustering!    ###########
##############################################################################
"""
    seleciona e executa todo o código marcardo por esta linha horizontal, porque senão 
    o plot não aparece no spider (estranho).
"""
######################################################################################
# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglomModel.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
    
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')



# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()
######################################################################################


##############################################################################
##############                Dendrogram the Model                 ###########
##############################################################################
"""
Dendrogram Associated for the Agglomerative Hierarchical Clustering

    Remember that a distance matrix contains the distance from each point to 
    every other point of a dataset.

    Use the function distance_matrix, which requires two inputs. Use the Feature 
    Matrix, X1 as both inputs and save the distance matrix to a variable called 
    dist_matrix

    Remember that the distance values are symmetric, with a diagonal of 0's. 
    This is one way of making sure your matrix is correct.
    (print out dist_matrix to make sure it's correct)
"""
dist_matrix = distance_matrix(X1,X1) 
print(dist_matrix)

"""
    Using the linkage class from hierarchy, pass in the parameters:
    
        The distance matrix
        'complete' for complete linkage
    
    
    A Hierarchical clustering is typically visualized as a dendrogram as shown 
    in the following cell. Each merge is represented by a horizontal line. The 
    y-coordinate of the horizontal line is the similarity of the two clusters 
    that were merged, where cities are viewed as singleton clusters. By moving 
    up from the bottom layer to the top node, a dendrogram allows us to 
    reconstruct the history of merges that resulted in the depicted clustering.
"""
Z = hierarchy.linkage(dist_matrix, 'complete')  # you can also run with 'average'
dendro = hierarchy.dendrogram(Z)












##############################################################################
#####             Hierarchical Clustering on a Vehicle Data Set            ###
##############################################################################
#############        Getting Data Set and pre-processing           ###########
##############################################################################
# !wget -O cars_clus.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/cars_clus.csv

filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)

pdf.head(5)


"""
    The feature sets include price in thousands (price), engine size (engine_s), 
    horsepower (horsepow), wheelbase (wheelbas), width (width), length (length), 
    curb weight (curb_wgt), fuel capacity (fuel_cap) and fuel efficiency (mpg).
    
    Lets simply clear the dataset by dropping the rows that have null value
"""
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)


"""
    Let us select the set of features we want to work with
"""
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

"""
    Now we can normalize the feature set. MinMaxScaler transforms features by 
    scaling each feature to a given range. It is by default (0, 1). That is, 
    this estimator scales and translates each feature individually such that it 
    is between zero and one.
"""
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
featureset_normalized = min_max_scaler.fit_transform(x)
featureset_normalized [0:5]



##############################################################################
#######               Creating Hierarchical Cluster Model                  ###
##############################################################################
# First, calculate the distance matrix
import scipy
leng = featureset_normalized.shape[0]
DistanceMatrix = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        DistanceMatrix[i,j] = scipy.spatial.distance.euclidean(featureset_normalized[i], featureset_normalized[j])
DistanceMatrix


"""
    In agglomerative clustering, at each iteration, the algorithm must update 
    the distance matrix to reflect the distance of the newly formed cluster 
    with the remaining clusters in the forest. The following methods are 
    supported in Scipy for calculating the distance between the newly formed 
    cluster and each:
        
        - single
        - complete
        - average
        - weighted
        - centroid

    We use complete for our case, but feel free to change it to see how the 
    results change.
"""
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(DistanceMatrix, 'complete')


"""
    Essentially, Hierarchical clustering does not require a pre-specified 
    number of clusters. However, in some applications we want a partition of 
    disjoint clusters just as in flat clustering. So you can use a cutting line:
"""
from scipy.cluster.hierarchy import fcluster
max_d = 3   # threshold to apply when forming flat clusters
clusters = fcluster(Z, max_d, criterion='distance')
clusters

"""
    Also, you can determine the number of clusters directly:
        
    from scipy.cluster.hierarchy import fcluster
    k = 5
    clusters = fcluster(Z, k, criterion='maxclust')
    clusters
"""

# Now, Plotting
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')














