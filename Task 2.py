#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business analytics internship

# # Task 2 - Prediction using Unsupervised ML

# In[3]:


#import the relevant Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import datasets


# In[4]:


#import Data
data_df = datasets.load_iris()
data = pd.DataFrame(data_df.data, columns = data_df.feature_names)

# take a look at the dataset

data.head()


# In[5]:


data.tail()


# # Data Exploration
# lets first have descriptive exploration of our data

# In[6]:


data.describe()


# # Data preprocessing

# Checking the null values

# In[7]:


data.isnull().sum()


# we can see there are no null values in the dataset

# # Checking for Outliers in each of the column by visualising on boxplot.

# In[8]:


sns.boxplot(data=data,x='sepal length (cm)')


# In[9]:


sns.boxplot(data=data,x='petal width (cm)')


# In[10]:


sns.boxplot(data = data,x ='petal length (cm)')


# In[11]:



sns.boxplot(data=data,x='sepal width (cm)')

From the above boxplot graphs we can see there are outliers only in sepal width. let's remove them.
# In[12]:


#Removing the outliers 

q3 = data['sepal width (cm)'].quantile(.75)
q1 = data['sepal width (cm)'].quantile(.25)
iqr = q3-q1
iqr


# In[13]:


upperrange = q3+1.5*iqr
bottomrange = q1-1.5*iqr
data2 = data[(data['sepal width (cm)']>bottomrange) & (data['sepal width (cm)']<upperrange)]
data2


# In[14]:


sns.boxplot(data=data2,x='sepal width (cm)')


# # Take advantage of the Elbow method

# So now we can see there is no outliers in Sepal width.

# In[18]:


x = data.iloc[:, [0, 1, 2, 3]].values


# In[21]:


wcss = []
# I have chosen to get solutions from 1 to 10 clusters;
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Check the result
wcss


# In[23]:


# Plot the number of clusters vs WCSS

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# # Explore clustering solutions and select the number of clusters

# In[37]:


# Fiddle with K (the number of clusters)

kmeans = KMeans(3)
# Fit the data
kmeans.fit(x)


# In[38]:


#Predict the clusters
identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[39]:


# Create a new data frame with the predicted clusters
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[47]:


# plot
plt.figure(figsize=(10,6))
plt.scatter(x[ identified_clusters== 0, 0], x[identified_clusters == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[identified_clusters == 1, 0], x[identified_clusters == 1, 1], 
            s = 100, c = 'green', label = 'Iris-versicolour')
plt.scatter(x[identified_clusters == 2, 0], x[identified_clusters == 2, 1],
            s = 100, c = 'orange', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





# In[ ]:




