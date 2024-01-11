#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import column_or_1d

df = pd.read_excel(r'D:\paper\datasetpr.xls')
df.head()


# In[2]:


X1=df.loc[:, [ 'Ws_2']].values
X2=df.loc[:, [ 'DirpWave_1','WDir_2']].values
X3=df.loc[:, ['Hs_pre']].values
y=df.loc[:,['Hs']].values


# In[3]:


from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler().fit(X1)
X1 = scaler1.transform(X1)
scaler2 = StandardScaler().fit(X2)
X2 = scaler2.transform(X2)
scaler3 = StandardScaler().fit(X3)
X3 = scaler3.transform(X3)
scaler4= StandardScaler().fit(y)
y = scaler4.transform(y)


# In[4]:


X=np.hstack((X1, X2,X3))


# In[5]:



X_test=X[20461:26307]
y_test=y[20461:26307].ravel()
X_train=np.concatenate((X[0:20460],X[26308:]), axis=0)
y_train=np.concatenate((y[0:20460],y[26308:]), axis=0).ravel()


# In[6]:


svr_rbf = SVR(kernel="rbf", C=5, gamma='scale', epsilon=0.01)


# In[7]:


md=svr_rbf.fit(X_train,y_train)


# In[8]:


y_pred=md.predict(X_test)


# In[9]:


md.score(X_train,y_train)


# In[10]:


md.score(X_test,y_test)


# In[11]:


from sklearn.metrics import mean_squared_error

# predict on the test set
y_pred = md.predict(X_test)

# calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print the RMSE
print("RMSE of the SVR(rbf) model is:", rmse)


# In[12]:


# calculate the SI
mean=np.mean(X_test)
SI=rmse/mean
# print the RMSE
print("SI of the SVR(rbf) model is:", SI)


# In[13]:


# calculate the bias
bias = np.mean(y_pred) - np.mean(y_test)

# print the bias
print("Bias of the SVR(rbf) model is:", bias)


# In[14]:


import scipy.stats as stats
y_test = y_test.ravel()
y_pred = y_pred.ravel()

r, p_value = stats.pearsonr(y_test, y_pred)

print("Pearson correlation coefficient of testing data:", r)


# In[15]:


r_squared = r**2

print("R squared:", r_squared)


# In[16]:


y_pred_train=md.predict(X_train)


# In[17]:



y_train = y_train.ravel()
y_pred_train = y_pred_train.ravel()

R_train, p_value = stats.pearsonr(y_pred_train, y_train)

print("Pearson correlation coefficient of training:", R_train)


# In[18]:


import matplotlib.pyplot as plt
y_test2=scaler4.inverse_transform(y_test.reshape(-1,1))
y_pred2=scaler4.inverse_transform(y_pred.reshape(-1,1))
plt.scatter(y_test2, y_pred2)

# set the x and y axis labels
plt.xlabel('observed significant wave height(m)')
plt.ylabel('predicted significant wave height(m)')

x=np.linspace(0,5,100)
y=x
plt.plot(x, y,color='k')
# display the plot
plt.show()

