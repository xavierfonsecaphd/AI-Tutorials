"""
    DBSCAN Algorithm
    
    Let's:
        Use DBSCAN to do Density based clustering
        Use Matplotlib to plot clusters

    Most of the traditional clustering techniques, such as k-means, hierarchical 
    and fuzzy clustering, can be used to group data without supervision. However, 
    when applied to tasks with arbitrary shape clusters, or clusters within 
    cluster, the traditional techniques might be unable to achieve good results. 
    That is, elements in the same cluster might not share enough similarity or 
    the performance may be poor. Additionally, Density-based Clustering locates 
    regions of high density that are separated from one another by regions of 
    low density. Density, in this context, is defined as the number of points 
    within a specified radius.

    In this section, the main focus will be manipulating the data and properties 
    of DBSCAN and observing the resulting clustering.
    
    Notice: For visualization of map, you need basemap package.
    if you dont have basemap install on your machine, you can use the following 
    line to install it
        !conda install -c conda-forge  basemap matplotlib==3.1 -y
    Notice: you maight have to refresh your page and re-run the notebook after 
    installation


Created on Sat May  1, 2021
@author: Dr. Eng. Xavier Fonseca
Based on: https://www.coursera.org/learn/machine-learning-with-python
"""

import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

##############################################################################
###########               Randomly Generate a Dataset              ###########
##############################################################################
"""
    The function below will generate the data points and requires these inputs:
    
        centroidLocation: Coordinates of the centroids that will generate the 
                         random data.
                         Example: input: [[4,3], [2,-1], [-1,4]]
                         
        numSamples: The number of data points we want generated, split over the 
                        number of centroids (# of centroids defined in centroidLocation)
                        Example: 1500
                        
        clusterDeviation: The standard deviation between the clusters. The larger 
                        the number, the further the spacing.
                        Example: 0.5
"""
def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

# Use createDataPoints with the 3 inputs and store the output into variables X and y.
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)




##############################################################################
#############                Building the Model                    ###########
##############################################################################
"""
    DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. 
    This technique is one of the most common clustering algorithms which works 
    based on density of object. The whole idea is that if a particular point 
    belongs to a cluster, it should be near to lots of other points in that 
    cluster.

    It works based on two parameters: Epsilon and Minimum Points
        Epsilon determine a specified radius that if includes enough number of 
                points within, we call it dense area
        
        minimumSamples determine the minimum number of data points we want in a 
                neighborhood to define a cluster.
"""
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
labels


##############################################################################
#############                Distinguishing outliers                ##########
##############################################################################
"""
    Distinguish outliers

    Lets Replace all elements with 'True' in core_samples_mask that are in the 
    cluster, 'False' if the points are outliers.
"""
# Firts, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels


##############################################################################
#############                  Visualizing Data                     ##########
##############################################################################
# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 2, len(unique_labels)))

# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)








##############################################################################
###########           Weather Station Clustering Dataset              ########
##############################################################################
"""
    DBSCAN is specially very good for tasks like class identification on a 
    spatial context. The wonderful attribute of DBSCAN algorithm is that it can 
    find out any arbitrary shape cluster without getting affected by noise. For 
    example, this following example cluster the location of weather stations in 
    Canada. DBSCAN can be used here, for instance, to find the group of stations 
    which show the same weather condition. As you can see, it not only finds 
    different arbitrary shaped clusters, can find the denser part of 
    data-centered samples by ignoring less-dense areas or noises.

    Let's start playing with the data. We will be working according to the 
    following workflow:

        1-Loading data
        2-Overview data
        3-Data cleaning
        4-Data selection
        5-Clusteing
"""
"""
!wget -O weather-stations20140101-20141231.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/weather-stations20140101-20141231.csv

   Let's use Stn_Name, Lat, Long, and P' 

Name in the table   	Meaning

Stn_Name            	Station Name
Lat                 	Latitude (North+, degrees)
Long                	Longitude (West - , degrees)
Prov                	Province
Tm                  	Mean Temperature (°C)
DwTm                	Days without Valid Mean Temperature
D                   	Mean Temperature difference from Normal (1981-2010) (°C)
Tx                   	Highest Monthly Maximum Temperature (°C)
DwTx                   	Days without Valid Maximum Temperature
Tn                  	Lowest Monthly Minimum Temperature (°C)
DwTn                	Days without Valid Minimum Temperature
S                   	Snowfall (cm)
DwS                 	Days without Valid Snowfall
S%N                 	Percent of Normal (1981-2010) Snowfall
P                   	Total Precipitation (mm)
DwP                 	Days without Valid Precipitation
P%N                 	Percent of Normal (1981-2010) Precipitation
S_G                 	Snow on the ground at the end of the month (cm)
Pd                     	Number of days with Precipitation 1.0 mm or more
BS                   	Bright Sunshine (hours)
DwBS                	Days without Valid Bright Sunshine
BS%                 	Percent of Normal (1981-2010) Bright Sunshine
HDD                 	Degree Days below 18 °C
CDD                 	Degree Days above 18 °C
Stn_No              	Climate station identifier (first 3 digits indicate drainage basin, last 4 characters are for sorting alphabetically).
NA                  	Not Available
"""
import csv
import pandas as pd
import numpy as np

filename='weather-stations20140101-20141231.csv'

#Read csv
pdf = pd.read_csv(filename)
pdf.head(5)


##############################################################################
#############            Pre-processing data set                   ###########
##############################################################################
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)


##############################################################################
####     Visualization of stations (data set) before building model        ###
##############################################################################
"""
    Read this to plot countries and map
    https://scitools.org.uk/cartopy/docs/v0.13/matplotlib/feature_interface.html?highlight=country
    
    or this one to learn how to create advanced maps with Cartopy
    https://scitools.org.uk/cartopy/docs/latest/matplotlib/advanced_plotting.html
"""
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

# set view to Canada
xlon=-140
ylon=-50
xlat=40
ylat=65
#I think it's only filtering the points within the Canadian View (should be all...)
pdf = pdf[(pdf['Long'] > xlon) & (pdf['Long'] < ylon) & (pdf['Lat'] > xlat) &(pdf['Lat'] < ylat)]
pdf['xm']= pdf['Long'].tolist()
pdf['ym'] =pdf['Lat'].tolist()


plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([xlon, ylon, xlat, ylat])   # set view to Canada
#ax.set_global()
#ax.stock_img()

# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(states_provinces, edgecolor='gray')

plt.show()


# Adding Points of the stations onto the map
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   ax.plot(row.xm, row.ym, markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()



##############################################################################
##########              Building Model based on Lat & Lon                   ##
##############################################################################
###       Clustering of stations based on their location - Lat& Lon         ##
##############################################################################

from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)

Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels # adding a column clusters to the data set

# data without outliers I think
realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)

set(labels)


##############################################################################
####                  Visualization of clustered stations                  ###
##############################################################################
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

# set view to Canada
xlon=-140
ylon=-50
xlat=40
ylat=65


plt.figure()
myMap = plt.axes(projection=ccrs.PlateCarree())
myMap.set_extent([xlon, ylon, xlat, ylat])   


# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

myMap.add_feature(cfeature.LAND)
myMap.add_feature(cfeature.BORDERS)
myMap.add_feature(cfeature.COASTLINE)
myMap.add_feature(states_provinces, edgecolor='gray')

plt.show()

colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Adding Points and clusters of the stations onto the map, with scatter plot
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    myMap.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))









"""
        Another example, let is cluster data now based on the stations' location,
        mean, max, and min Temperature
"""
##############################################################################
##########                   Building Model based on                        ##
##############################################################################
###                Lat& Lon, mean, max, and min Temperature                 ##
##############################################################################
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)

##############################################################################
####                  Visualization of clustered stations                  ###
##############################################################################
###                Lat& Lon, mean, max, and min Temperature                 ##
##############################################################################
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

# set view to Canada
xlon=-140
ylon=-50
xlat=40
ylat=65


plt.figure()
myNewMap = plt.axes(projection=ccrs.PlateCarree())
myNewMap.set_extent([xlon, ylon, xlat, ylat])   


# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

myNewMap.add_feature(cfeature.LAND)
myNewMap.add_feature(cfeature.BORDERS)
myNewMap.add_feature(cfeature.COASTLINE)
myNewMap.add_feature(states_provinces, edgecolor='gray')


colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Adding Points and clusters of the stations onto the map, with scatter plot
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    myNewMap.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))












