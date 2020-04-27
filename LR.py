#!/usr/bin/env python
# coding: utf-8

# ## EXERCISE1
# ### Importing the librarys 

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df=pd.read_csv('~/documents/homeprices.csv')


# In[3]:


df


# ## data preprocessing 
# - fill na values with median value of the column

# In[4]:


df.bedrooms.median()


# In[5]:


df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
df


# In[6]:


reg=linear_model.LinearRegression()


# In[7]:


reg.fit(df.drop('price',axis='columns'),df.price)


# In[8]:


reg.coef_


# In[9]:


reg.intercept_


# ### Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old

# In[10]:


reg.predict([[3000,3,40]])


# ## Exercise 2

# 
# This file contains hiring statics for a firm such as experience of candidate, his written test score and personal interview score. Based on these 3 factors, HR will decide the salary. Given this data, you need to build a machine learning model for HR department that can help them decide salaries for future candidates. Using this predict salaries for following candidates,
# 
# - 2 yr experience, 9 test score, 6 interview score
# 
# - 12 yr experience, 10 test score, 10 interview score
# 
# 

# ### importing libraries

# In[ ]:





# In[12]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[13]:


df=pd.read_csv('~/documents/hiring.csv')


# In[14]:


df


# In[15]:


df.experience=df.experience.fillna('zero')


# In[16]:


df


# In[18]:


df.experience=df.experience.apply(w2n.word_to_num)


# In[19]:


df


# In[21]:


import math
median_test_score=math.floor(df['test_score(out of 10)'].mean())


# In[22]:


median_test_score


# In[23]:


df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(median_test_score)


# In[24]:


df


# In[25]:


reg=linear_model.LinearRegression()


# In[27]:


reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[28]:


reg.predict([[2,9,6]])


# In[29]:


reg.predict([[12,10,10]])


# In[ ]:




