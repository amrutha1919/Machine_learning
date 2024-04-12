#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


df_train = pd.read_csv('C:\\Users\\abhij\\Downloads\\Blood_samples_dataset_balanced_2(f).csv\\Blood_samples_dataset_balanced_2(f).csv')
df_test = pd.read_csv('C:\\Users\\abhij\\Downloads\\blood_samples_dataset_test.csv')


# In[6]:


df_train.head(10)


# In[7]:


df_train.dtypes


# In[8]:


df_test.head(10)


# In[9]:


df_test.dtypes


# In[12]:


df_train.shape


# In[13]:


df_test.shape


# In[14]:


df = pd.concat([df_train, df_test], ignore_index=True)


# In[15]:


df.shape


# In[16]:


X = df.drop(columns=['Disease'])
y = df['Disease']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)


# In[18]:


y_train.value_counts(), y_test.value_counts()


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


rf_model = RandomForestClassifier()


# In[21]:


rf_model.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[23]:


y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_rf, average='weighted')}")


# In[ ]:




