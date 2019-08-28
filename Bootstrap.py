#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as p
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math as math


# In[2]:


data = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/Boston_Housing_Data.csv")


# In[3]:


data


# In[12]:


y = data.values[:,13]
x = data.values[:,0:12]


# In[22]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)


# In[57]:


model = RandomForestRegressor(n_estimators = 500, min_samples_split = 40, max_features= None)


# In[58]:


model.fit(x_train,y_train)


# In[59]:


model.score(x_train,y_train)


# In[60]:


pred = model.predict(x_train)


# In[61]:


pred


# In[62]:


res = y_train-pred
res


# In[63]:


res_sq = res**2
mse=res_sq.mean()
mse


# In[64]:


rmse= math.sqrt(mse)
rmse


# In[65]:


predtest = model.predict(x_test)


# In[66]:


model.score(x_test,y_test)


# In[67]:


restest = y_test - predtest


# In[68]:


restest


# In[69]:


restest_sq = restest**2
testmse=restest_sq.mean()
testmse


# In[70]:


testrmse= math.sqrt(testmse)
testrmse 


# In[ ]:





# In[ ]:





# In[ ]:




