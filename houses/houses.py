
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier
import lightgbm as lgb


# In[2]:


train_path = 'train.csv'
test_path = 'test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# In[3]:


big_data = pd.concat([train_data, test_data])


# In[4]:


# columns = list(range(1,14)) + [-1]
# trans_train = train_data.iloc[:,columns].dropna()
trans_train = big_data

dt = pd.to_datetime(trans_train['timestamp'])
trans_train['timestamp'] = dt.astype(np.int64) // 10 ** 9
# pd.Series(trans_train['sub_area']).unique()
trans_train['sub_area'] = 'sub_area_' + trans_train['sub_area']
# new_cols = pd.Series(trans_train['sub_area']).unique()
one_hots = pd.get_dummies(pd.Series(trans_train['sub_area']))
trans_train=pd.concat([trans_train,one_hots],axis=1)

yes_no_cols = ['thermal_power_plant_raion','incineration_raion','oil_chemistry_raion',
               'radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion',
               'detention_facility_raion','big_road1_1line','railroad_1line','water_1line']

yes_no_dict = {'yes':1, 'no':0}
[trans_train[item].replace(yes_no_dict, inplace=True) for item in yes_no_cols]

ecology_dict = {'good':2, 'excellent':3, 'poor':0, 'satisfactory':1, 'no data':1}
trans_train['ecology'].replace(ecology_dict, inplace=True)

prod_dict = {'Investment':0, 'OwnerOccupier':1}
trans_train['product_type'].replace(prod_dict, inplace=True)

bad_cols = ['culture_objects_top_25', 'sub_area']
trans_train.drop(bad_cols,axis=1,inplace=True)


# print(trans_train.dtypes[trans_train.dtypes == 'object'])


# In[5]:


y_train = trans_train[:len(train_data)]['price_doc'].reshape(-1,1)

trans_train.dropna(axis=1, inplace=True)
x_train = trans_train[:len(train_data)]
x_test = trans_train[len(train_data):]


# In[ ]:


# lgb_train = lgb.Dataset(x_train, y_train)
# # lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# # specify your configurations as a dict
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': {'l2', 'auc'},
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }

# print('Start training...')
# # train
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=20,
# #                 valid_sets=lgb_eval,
#                 early_stopping_rounds=5)



lr_mod = linear_model.LinearRegression()
lr_mod.transform = lr_mod.predict
xgb = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)

# pipe = Pipeline([('lr', lr_mod), ('xgb', XGBClassifier())])
# pipe = Pipeline([ ('xgb', xgb)])
xgb.fit(x_train, y_train)


# In[ ]:


y_test = pipe.predict(x_test)


# In[ ]:


x_test['price_doc'] = y_test


# In[ ]:


x_test[['id', 'price_doc']].to_csv('soln.csv', index=False)

