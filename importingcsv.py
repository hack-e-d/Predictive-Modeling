#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as p;


# In[3]:


data = p.read_csv("C:/Users/COE-20/Desktop/Python/Data/Credit_Card_Expenses.csv");


# In[9]:


data 


# In[11]:


cc =data.CC_Expenses
cc


# In[14]:


cc.mean()


# In[15]:


cc.median()


# In[16]:


cc.std()


# In[17]:


cc.var()


# In[18]:


cc.min()


# In[20]:


cc.quantile(0.99)


# In[21]:


cc.describe()


# In[23]:


import matplotlib.pyplot as ploter


# In[24]:


ploter.hist(cc);
ploter.show();


# In[25]:


ploter.boxplot(cc);
ploter.show();


# In[ ]:




