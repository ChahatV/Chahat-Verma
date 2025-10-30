#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("orders.csv", na_values = ['Not Available','unknown'])
#handle null values


# In[2]:


df['Ship Mode'].unique()


# In[3]:


# rename col names - lowerCase and replace space with underscore -- used str.lower and str.replace
#df.rename(columns = {'Order Id' : 'order_id', 'City' : 'city'})

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')
df.columns


# In[4]:


#derive new columns discount, sale price and profit

df['discount']= df['list_price']*df['discount_percent']*.01


# In[5]:


df['sales_price'] = df['list_price'] - df['discount']
df


# In[6]:


df['profit'] = df['cost_price'] - df['sales_price']


# In[7]:


df['order_date'] = pd.to_datetime(df['order_date'], format = "%Y-%m-%d")


# In[8]:


#drop costprice, list price and discount percent cols
df.drop(columns=['list_price','discount_percent','cost_price'], inplace = True)


# In[14]:


pip install psycopg2-binary


# In[ ]:





# In[24]:


from sqlalchemy import create_engine, text

# ✅ Correct connection string (with URL-encoded '@')
engine = create_engine(
    "postgresql+psycopg2://postgres:chahat1712@localhost:5433/postgres"
)

conn = engine.connect()


# In[25]:


#load the data in sql server

df.to_sql('df_orders', con = conn, index = False, if_exists = 'replace')


# In[26]:


df.head(10)


# In[ ]:





# In[ ]:




