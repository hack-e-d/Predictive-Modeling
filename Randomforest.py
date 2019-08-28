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


# In[39]:


model = RandomForestRegressor(n_estimators = 500, min_samples_split = 40, max_features= "auto")


# In[40]:


model.fit(x_train,y_train)


# In[41]:


model.score(x_train,y_train)


# In[42]:


pred = model.predict(x_train)


# In[43]:


pred


# In[44]:


res = y_train-pred
res


# In[45]:


res_sq = res**2
mse=res_sq.mean()
mse


# In[46]:


rmse= math.sqrt(mse)
rmse


# In[47]:


predtest = model.predict(x_test)


# In[48]:


model.score(x_test,y_test)


# In[49]:


restest = y_test - predtest


# In[50]:


restest


# In[51]:


restest_sq = restest**2
testmse=restest_sq.mean()
testmse


# In[56]:


testrmse= math.sqrt(testmse)
testrmse 


# In[ ]:





# In[ ]:





# In[ ]:




