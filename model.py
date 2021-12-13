#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard Libraries: 
import pandas as pd
import numpy as np 

# Visuals:
import matplotlib.pyplot as plt
import seaborn as sns

# Stats:
from scipy import stats

# Splitting
from sklearn.model_selection import train_test_split

# Modeling
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import TweedieRegressor
import sklearn.preprocessing

# My Files
from env import host, user, password
import wrangle
import explore

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# comment out viz before putting in report
train, validate, test = wrangle.wrangle_zillow()


# In[3]:


train.shape, validate.shape, test.shape


# In[4]:


# Run split_tvt_into_variables / the target is tax_value
train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = wrangle.split_tvt_into_variables(train, validate, test, target='tax_value')


# In[5]:


scaler, X_train_scaled, X_validate_scaled, X_test_scaled = wrangle.Min_Max_Scaler(X_train, X_validate, X_test)


# Baseline: 

# In[55]:


def baseline(y_train, y_validate):
    
    # can't add new predicted column to Y until we turn it into a dataframe 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. Predict median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median

    # 3. RMSE of predicted mean
    rmse_train = mean_squared_error(y_train.tax_value,
                                y_train.value_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_mean) ** (0.5)
    
    # 4. RMSE of predicted median
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_median) ** .5
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_median) ** (0.5)

    # building a df of our metrics for model selsection 
    metric_df = pd.DataFrame(data=[
            {
                'model': 'mean_baseline', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    return metric_df
#metric_df


# In[108]:


# baseline(y_train, y_validate)


# In[87]:


def OLS(X_train_scaled, X_validate_scaled, y_train, y_validate):
    
    # can't add new predicted column to Y until we turn it into a dataframe 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. Predict median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median
    
    # create lm model
    lm = LinearRegression()
    
    # fit model
    lm.fit(X_train_scaled, y_train.tax_value)
    
    #predict train
    y_train['value_pred_lm'] = lm.predict(X_train_scaled)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm) ** (1/2)
    
    # predict validate
    y_validate['value_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm) ** (1/2)
    
    # building a df of our metrics for model selsection 
    metric_df = pd.DataFrame(data=[
            {
                'model': 'OLS Regressor', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    return metric_df


# In[109]:


# OLS(X_train_scaled, X_validate_scaled, y_train, y_validate)


# In[113]:


def Lasso_Lars(X_train_scaled, X_validate_scaled, y_train, y_validate):
    
    # can't add new predicted column to Y until we turn it into a dataframe 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. Predict median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median
    
    # create
    lars = LassoLars(alpha=0.01)

    # fit
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lars) ** (1/2)

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lars) ** (1/2)

    # building a df of our metrics for model selsection 
    metric_df = pd.DataFrame(data=[
            {
                'model': 'lasso_alpha0.01', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    return metric_df    


# In[110]:


# Lasso_Lars(X_train_scaled, X_validate_scaled, y_train, y_validate)


# In[91]:


def Linear_Regression(X_train_scaled, X_validate_scaled, y_train, y_validate):
    # can't add new predicted column to Y until we turn it into a dataframe 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. Predict median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median
    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 =  pf.transform(X_test_scaled)
    
    # create
    lm2 = LinearRegression()

    # fit
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm2) ** (1/2)

    # predict validate
    y_validate['value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm2) ** 0.5

    # building a df of our metrics for model selsection 
    metric_df = pd.DataFrame(data=[
            {
                'model': 'linear_regression', 
                'RMSE_train': rmse_train,
                'RMSE_validate': rmse_validate
                }
            ])
    return metric_df 


# In[23]:


# Linear_Regression(X_train_scaled, X_validate_scaled, y_train, y_validate)


# In[17]:


def Lasso_Lars_Viz(X_train_scaled, X_validate_scaled, y_train, y_validate):
    
    # can't add new predicted column to Y until we turn it into a dataframe 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. Predict median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median
    
    # create
    lars = LassoLars(alpha=0.01)

    # fit
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lars) ** (1/2)

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lars) ** (1/2)

    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.tax_value, color='blue', alpha=.5, label="Actual Tax Value")
    plt.hist(y_validate.value_pred_lars, color='red', alpha=.5, label="Model: LassoLars")
    plt.xlabel("Tax Value")
    plt.ylabel("Amount of Houses")
    plt.title("Comparing the Distribution of Actual Values to Distributions of Predicted Values for the Top Model-LassoLars")
    plt.legend()
    return plt.show()


# In[112]:


# Lasso_Lars_Viz(X_train_scaled, X_validate_scaled, y_train, y_validate)


# In[33]:


def Lasso_Lars_Test(X_train_scaled, X_validate_scaled, y_train, y_validate, X_test_scaled, y_test):
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict value_pred_mean
    value_pred_mean = y_train.tax_value.mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean
    
    # 2. compute value_pred_median
    value_pred_median = y_train.tax_value.median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['_pred_median'] = value_pred_median
    
    # 3. RMSE of value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value,
                                y_train.value_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_mean) ** (0.5)
    
    # create
    lars = LassoLars(alpha=0.01)

    # fit
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lars) ** (1/2)

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lars) ** (1/2)

    
    y_test = pd.DataFrame(y_test)

    # predict on test
    y_test['value_pred_lars'] = lars.predict(X_test_scaled)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.value_pred_lars) ** (1/2)                 

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nTest/Out-of-Sample: ", rmse_test,
      "\nDifference: ", rmse_test - rmse_train)
    
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.tax_value, color='blue', alpha=.5, label="Actual Tax Value")
    plt.hist(y_validate.value_pred_lars, color='red', alpha=.5, label="Model: LassoLars")
    plt.xlabel("Tax Value")
    plt.ylabel("Amount of Houses")
    plt.title("Comparing the Distribution of Actual Values to Distributions of Predicted Values for the Top Model-LassoLars")
    plt.legend()
    return plt.show()


# In[35]:


# Lasso_Lars_Test(X_train_scaled, X_validate_scaled, y_train, y_validate, X_test_scaled, y_test)


# In[37]:


# We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
y_train = pd.DataFrame(y_train)
y_validate = pd.DataFrame(y_validate)


# In[38]:


# 1. Predict vlue_pred_mean
value_pred_mean = y_train.tax_value.mean()
y_train['value_pred_mean'] = value_pred_mean
y_validate['value_pred_mean'] = value_pred_mean


# In[39]:


# 2. compute value_pred_median
value_pred_median = y_train.tax_value.median()
y_train['value_pred_median'] = value_pred_median
y_validate['_pred_median'] = value_pred_median


# In[40]:


# 3. RMSE of value_pred_mean
rmse_train = mean_squared_error(y_train.tax_value,
                                y_train.value_pred_mean) ** .5
rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_mean) ** (0.5)


# In[21]:


# create
lars = LassoLars(alpha=0.01)

# fit
lars.fit(X_train_scaled, y_train.tax_value)

# predict train
y_train['value_pred_lars'] = lars.predict(X_train_scaled)

# evaluate: rmse
rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lars) ** (1/2)

# predict validate
y_validate['value_pred_lars'] = lars.predict(X_validate_scaled)

# evaluate: rmse
rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lars) ** (1/2)

#print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
#      "\nValidation/Out-of-Sample: ", rmse_validate,
#     "\nDifference: ", rmse_validate - rmse_train)


# In[22]:


y_test = pd.DataFrame(y_test)

# predict on test
y_test['value_pred_lars'] = lars.predict(X_test_scaled)

# evaluate: rmse
rmse_test = mean_squared_error(y_test.tax_value, y_test.value_pred_lars) ** (1/2)

#print("RMSE for OLS Model using LinearRegression\nOut-of-Sample Performance: ", rmse_test)


# In[20]:


# plot to visualize actual vs predicted. 
#plt.figure(figsize=(16,8))
#plt.hist(y_validate.tax_value, color='blue', alpha=.5, label="Actual Tax Value")
#plt.hist(y_validate.value_pred_lars, color='red', alpha=.5, label="Model: Lasso_Lars")
#plt.xlabel("Tax Value")
#plt.ylabel("Amount of Houses")
#plt.title("Comparing the Distribution of Actual Values to Distributions of Predicted Values for the Top Model-LassoLars")
#plt.legend()
#plt.show()


# In[ ]:





# In[ ]:




