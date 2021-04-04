#!/usr/bin/env python
# coding: utf-8

# # Data science and Business Analytics Internship
# 

# # Task 1 ( Prediction using supervised ML)
# 

# In[21]:


#import the relevant libraries

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the data

# In[22]:



data = pd.read_csv('student_scores.csv')


# In[23]:


# take a look at the dataset

data.head() 


# # Data Exploration
# 
# lets first have descriptive exploration of our data

# In[24]:


data.describe()


# In[25]:



data.hist()
plt.show()


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[26]:


plt.scatter(data['Hours'],data['Scores'], c = "green")
plt.xlabel("No. of Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# # Data preprocessing

# In[27]:


#Checking for null values

data.isnull().sum()


# In[52]:


#Normalization of data using simple feature scaling

data["Hours"] = data["Hours"]/data["Hours"].max()
data["Scores"] = data["Scores"]/data["Scores"].max()
data.head()


# # Preparing the data
# 
# Creating train and test dataset

# In[53]:


msk = np.random.rand(len(data)) < 0.8  #select random rows using np.random.rand() function
train = data[msk]
test = data[~msk]


# Train data distribution
# 

# In[54]:


plt.scatter(train.Hours,train.Scores, c = "green")
plt.xlabel("No. of Hours studied")
plt.ylabel("Percentage Scores")
plt.show()


# # Modeling

# In[55]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Hours']])
train_y = np.asanyarray(train[['Scores']])
regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Plot Outputs
# 

# In[56]:


plt.scatter(train.Hours, train.Scores,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("No of Hours Studied")
plt.ylabel("Percentage score")


# # Making Predictions

# In[57]:


test_y_hat = regr.predict(test_x)
test_y_hat


# In[58]:


#What will be predicted score if a student studies for 9.25 hrs/ day?

own_pred = regr.intercept_ + regr.coef_*9.25
print("If a student studies 9.25 hrs/day predicted scores will be", own_pred)


# # Evaluation

# In[59]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Hours']])
test_y = np.asanyarray(test[['Scores']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[ ]:




