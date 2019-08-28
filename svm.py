#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as p
from sklearn import svm


# In[11]:


data = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/Iris_data.csv")


# In[12]:


data


# In[13]:


x = data.values[:, 0:4]
y = data.values[:, 4]


# In[14]:


mymodel = svm.SVC() 
mymodel.fit(x, y)
mymodel.score(x, y)
pred = mymodel.predict(x)
mytable = p.crosstab(y, pred)
mytable


# In[15]:


mytestdata = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/Iris_test.csv")
test_x = mytestdata.values[:, 0:4]
test_y = mytestdata.values[:, 4]
pred_test = mymodel.predict(test_x)
mytesttable = p.crosstab(test_y, pred_test)
mytesttable


# In[ ]:





# In[ ]:




