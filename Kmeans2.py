# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:34:11 2016

@author: Erin
"""
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import pandas as pd
import collections
import numpy as np
import scipy.cluster
from scipy.cluster import vq


#get data
data = pd.read_csv('C:/Users/Erin/thinkful/Unit4Lesson5/un_new.csv')
data = pd.DataFrame(data)
print data[0:10]

datasize = len(data)
print "datasize ="
print datasize

#number of rows that are not null
for i in data:
    print 'Size of column without NaN is:' 
    print (i, len(data[i].dropna()))

#number of countries in the dataset
print 'The number of countries in the dataset = %s' % len(collections.Counter(data['country']))


#data type of each column
for i in data:
    print 'Data type of column %s is %s' % (i, data[i].dtype)


# Drop NaN 
data = data.dropna()
print data[0:10]

#and convert dataframe into array
array = pd.DataFrame.as_matrix(data, columns = ['lifeMale', 'lifeFemale', \
                                'infantMortality', 'GDPperCapita'])
print array

#explore 1 to 10 clusters
# Normalize data
whitened = vq.whiten(array)

# Finding numbers of k / clusters between 1 - 10
centroids = {}
for k in range(1,11):
    centers,dist = vq.kmeans(whitened, k)
    code, distance = vq.vq(whitened, centers)
    distance = np.array(distance).tolist()
    centroids[k] = distance
    
for k, v in centroids.iteritems():
    ss = 0                          # sum of squares
    for i in v:
        ss += i**2
    centroids[k] = round(ss / len(v), 1)
#getting error "ValueError: low >= high"
#Github Wiharto for printing   
#print centroids
x_axis = []
y_axis = []
for k, v in centroids.iteritems():
    x_axis.append(k)
    y_axis.append(v)
    
plt.plot(x_axis, y_axis, '-ro')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Within-Cluster Sum of Squares')
plt.show()

#Plot the number of clusters against the average within-cluster sum of squares
## K = 3
centers, dist = vq.kmeans(whitened, 3)
code, distance = vq.vq(whitened, centers)

#print centers
#print dist
#print code
#print distance

a = array[code==0]
b = array[code==1]
c = array[code==2]

# Plot 'infant Mortality' VS 'GDP per Capita'
plt.figure(figsize = (8,6))
plt.scatter(a[:,3], a[:,2], c = 'g')
plt.scatter(b[:,3], b[:,2], c = 'r')
plt.scatter(c[:,3], c[:,2], c = 'b')
plt.xlabel('GDP per Capita')
plt.ylabel('Infant Mortality')
plt.plot(centers[:, 2], centers[:, 3], 'bx', markersize = 8)
plt.show()

# Plot 'lifeMale' VS 'GDP per Capita'
plt.figure(figsize = (8,6))
#plt.scatter(a[:,0], a[:,3], c = 'g')
#plt.scatter(b[:,0], b[:,3], c = 'r')
#plt.scatter(c[:,0], c[:,3], c = 'b')
plt.plot(a[:,3], a[:,0], 'go')
plt.plot(b[:,3], b[:,0], 'ro')
plt.plot(c[:,3], c[:,0], 'bo')
plt.xlabel('GDP per Capita')
plt.ylabel('Life Expectancy for Male')
plt.show()

# Plot 'lifeFemale' VS 'GDP per Capita'
plt.figure(figsize = (8,6))
plt.plot(a[:,3], a[:,1], 'go')
plt.plot(b[:,3], b[:,1], 'ro')
plt.plot(c[:,3], c[:,1], 'bo')
plt.xlabel('GDP per Capita')
plt.ylabel('Life Expectancy for Female')
plt.show()




