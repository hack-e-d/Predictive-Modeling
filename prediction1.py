#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as p;


# In[2]:


from sklearn import tree; 


# In[3]:


data = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/Mail_Respond.csv");


# In[4]:


data


# In[6]:


y= data.Outcome;
x= data[["District","House_Type","Income","Previous_Customer"]]


# In[7]:


model =tree.DecisionTreeClassifier(min_samples_split = 10);


# In[10]:


model.fit(x,y)


# In[11]:


pred = model.predict(x)


# In[12]:


x


# In[13]:


pred


# In[14]:


y


# In[15]:


table = p.crosstab(y,pred)
table


# In[16]:


model.score(x,y)


# In[ ]:




