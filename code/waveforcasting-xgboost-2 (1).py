#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

X_test=X[20461:26307]
y_test=y[20461:26307].ravel()
X_train=np.concatenate((X[0:20460],X[26308:]), axis=0)
y_train=np.concatenate((y[0:20460],y[26308:]), axis=0).ravel()


# In[5]:


xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.5,
    colsample_bytree=0.8,
    early_stopping_rounds=3
)
xgbmd=xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
xgb_model.score(X_test, y_test)


# In[6]:


xgbmd.score(X_train, y_train)


# In[7]:


from sklearn.metrics import mean_squared_error

# predict on the test set
y_pred = xgbmd.predict(X_test)

# calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print the RMSE
print("RMSE of the XGBoost model is:", rmse)


# In[8]:


# calculate the SI
mean=np.mean(X_test)
SI=rmse/mean
# print the RMSE
print("SI of the XGBoost model is:", SI)


# In[9]:


# calculate the bias
bias = np.mean(y_pred) - np.mean(y_test)

# print the bias
print("Bias of the XGBoost model is:", bias)


# In[10]:


import scipy.stats as stats
y_test = y_test.ravel()
y_pred = y_pred.ravel()

r, p_value = stats.pearsonr(y_test, y_pred)

print("Pearson correlation coefficient:", r)


# In[11]:


y_pred_train=xgbmd.predict(X_train)


# In[12]:


y_train = y_train.ravel()
y_pred_train = y_pred_train.ravel()

R_train, p_value = stats.pearsonr(y_pred_train, y_train)

print("Pearson correlation coefficient of training:", R_train)


# In[ ]:





# In[13]:


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

