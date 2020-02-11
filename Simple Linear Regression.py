#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


salary = pd.read_csv('/Users/KD/Desktop/Salary_Data.csv')


# In[9]:


salary.head()


# In[13]:


#devide dataset into x and y. X is independent variable and y is dependent 
X = salary.iloc[:, :-1].values #remove salary column which is a dependent variable
y = salary.iloc[:, 1].values #only keep salary column


# In[16]:


from sklearn.model_selection import train_test_split
#split data in train and test 


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 1/3, random_state = 0)


# In[28]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train, y_train)


# In[29]:


# Predicting the Test data set results
y_pred = regressor.predict(X_test)


# In[32]:


#implememt graph simple linear regresssion
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, simpleLinearRegression.predict(X_train), color = 'blue')


# In[33]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, simpleLinearRegression.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

