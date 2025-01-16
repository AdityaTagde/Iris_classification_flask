#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('darkgrid')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris=load_iris()


# In[4]:


iris


# In[5]:


ds=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[6]:


ds


# In[7]:


ds['label']=iris.target


# In[8]:


ds


# In[9]:


X=ds.iloc[:,0:4].values


# In[10]:


y=ds.iloc[:,4].values


# In[11]:


X


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


model=DecisionTreeClassifier(max_leaf_nodes=4,max_features=3,max_depth=3)


# In[19]:


model.fit(X_train,y_train)


# In[20]:


y_pred=model.predict(X_test)


# In[21]:


y_pred


# In[22]:


y_test


# In[23]:


from sklearn.metrics import confusion_matrix,classification_report


# In[24]:


print(confusion_matrix(y_test,y_pred))


# In[25]:


print(classification_report(y_test,y_pred))


# In[27]:


import pickle 

with open('irisfsk.pkl','wb') as f: 
    pickle.dump(model,f)


# In[ ]:




