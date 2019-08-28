#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as p
from sklearn import tree
from sklearn.cross_validation import train_test_split


# In[6]:


data = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/bank-data.csv")


# In[7]:


data


# In[4]:


y= data.pep


# In[9]:


x= data.values[:,0:9]


# In[10]:


x


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 30)


# In[19]:


model = tree.DecisionTreeClassifier(min_samples_split = 50)


# In[20]:


model.fit(x_train,y_train)


# In[21]:


model.score(x_train,y_train)


# In[22]:


pred = model.predict(x_test)


# In[23]:


table = p.crosstab(y_test,pred)
table


# In[ ]:




