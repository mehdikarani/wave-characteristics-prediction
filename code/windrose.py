#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from windrose import WindroseAxes

df = pd.read_excel(r'D:\papare\datasetpr.xls')
df.head()


# In[2]:


ax = WindroseAxes.from_ax()
ax.bar(df['WDir_2'], df['Ws_2'], normed=True, opening=0.8, edgecolor='white')
ax.set_legend()


# In[3]:


plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
df2 = pd.read_excel(r'D:\papare\d2L.xlsx')
df2.head()


# In[5]:


X=df2.loc[:, ['dir']].values
y=df2.loc[:,['l']].values.ravel()


# In[6]:


xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.5,
    colsample_bytree=0.8,
)
xgb_model.fit(X, y)


# In[17]:


di=df.loc[:, ['WDir_2']].values
di = di.flatten()


# In[18]:


fechL=xgb_model.predict(di)


# In[28]:


data={
    'dir': di,
    'L': fechL,
}
res = pd.DataFrame(data)
res.to_excel(r'D:\papare\results.xlsx', index=False)


# In[26]:




