#!/usr/bin/env python
# coding: utf-8

# # Project Name - House Prices: Advanced Regression Techniques
# The main aim of this project is to predict the house price based on various features which have been provided in the data set which can be downloaded from the link below
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# In[734]:


get_ipython().run_line_magic('cd', '/Users/muditg19/downloads/DOWNLOADS/house-prices-advanced-regression-techniques')


# In[735]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[736]:


df=pd.read_csv('train.csv')
df.shape


# In[737]:


df.isnull().sum()


# In[738]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[739]:


#Seaching for categorical missing values
features_nan=[feature for feature in df.columns if df[feature].isnull().sum()>=1 and df[feature].dtypes=='O']
for feature in features_nan:
    print("{} - {}% missing values".format(feature,np.round(df[feature].isnull().mean()*100,4)))


# In[740]:


## droping all the features having more than 75% of missing data
for feature in features_nan:
    if (np.round(df[feature].isnull().mean(),4)*100)>75.0:
        df.drop([feature], axis=1, inplace=True)
        features_nan.remove(feature)


# In[741]:


df.head()


# In[742]:


## Replace missing values with a 'Missing' label
def replace_catg_feature(df,features_nan):
    data=df.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

df=replace_catg_feature(df,features_nan)


# In[743]:


df[features_nan].isnull().sum()


# In[744]:


df.head()


# In[745]:


## checking for numerical variables that contains missing values
num_var_with_nan=[feature for feature in df.columns if df[feature].isnull().sum()>=1 and df[feature].isnull().sum()!='O']


# In[746]:


for feature in num_var_with_nan:
    print("{}: {}% missing value".format(feature,np.around(df[feature].isnull().mean()*100,4)))


# In[747]:


## Replacing the numerical Missing Values

for feature in num_var_with_nan:
    median_value=df[feature].median()
    
    ## create a new feature to capture nan values
    df[feature+'nan']=np.where(df[feature].isnull(),1,0)
    df[feature].fillna(median_value,inplace=True)
    
df[numerical_with_nan].isnull().sum()


# In[748]:


df.head(50)


# In[749]:


# Exploring Temporal Data
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']:
    print(feature, df[feature].unique())


# In[750]:


## Checking the relationship between year the house is sold and the sales price
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[751]:


## The above graph dosen't show the right information about the data

## Comparing the difference between All years feature with SalePrice to get better info about the data
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']:
    if feature!='YrSold':
        data=df.copy()
        ## capturing the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[752]:


# These graphs suggest that the older the year feature the lesser the price
# Preprocessing temporal Variable
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       df[feature]=df['YrSold']-df[feature]


# In[753]:


## Exploring Numerical variables 
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# In[754]:


## checking both Continous variable and Discrete Variables
discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25 and feature not in ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','LotFrontagenan',
 'MasVnrAreanan','GarageYrBltnan','Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold','LotFrontagenan',
 'MasVnrAreanan','GarageYrBltnan','Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[755]:


print(discrete_feature)
df[discrete_feature].head()


# In[756]:


print(continuous_feature)
df[continuous_feature].head()


# In[757]:


## Relationship between discrete features and Sale price
for feature in discrete_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[758]:


#majority of them show exponential graph suggesting monotonic realtionship


# In[759]:


## Relationship between Continous features and Sale price
for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[760]:


#Except for a very few all of them don't show gaussian distribution impling they have skewed data
# using Logarithimic transformation to get a increase the data quality

import numpy as np
num_features=[feature for feature in continuous_feature if 0 not in data[feature].unique()] #Considering only those which does not have zero value inside
for feature in num_features:
    df[feature]=np.log(df[feature])
    
for feature in num_features:
    data=df.copy()
    data[feature]=np.log(data[feature])
    data['SalePrice']=np.log(data['SalePrice'])
    plt.scatter(data[feature],data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalesPrice')
    plt.title(feature)
    plt.show()


# In[761]:


# Visualising outliers
for feature in num_features:
    data[feature]=np.log(data[feature])
    data.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# In[762]:


# preprocessing categorical data
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']
categorical_features


# In[763]:


## Find out the relationship between categorical variable and dependent feature SalesPrice

for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[764]:


for feature in categorical_features:
    temp=df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index ##categories in cat var which are present in less than 1% of the observations
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var') # replacing those categories with "Rare_var" 


# In[765]:


df.head(50)


# In[766]:


for feature in categorical_features:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)


# In[767]:


scaling_feature=[feature for feature in df.columns if feature not in ['Id','SalePerice'] ]


# In[768]:


feature_scale=[feature for feature in df.columns if feature not in ['Id','SalePrice']]
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
scaler.fit(df[feature_scale])


# In[769]:


scaler.transform(df[feature_scale])


# In[770]:


df.head()


# In[791]:


# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([df[['SalePrice','Id']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)],
                    axis=1)


# In[785]:


#splitting the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.drop(["SalePrice"],axis=1), data['SalePrice'], test_size = 0.2, random_state=0)


# In[772]:


data.to_csv('X_train.csv',index=False)


# In[792]:


#Linear regression model
from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[793]:


#fitting linear regression on the data
model = lr.fit(X_train, y_train)


# In[794]:


#R square value
print('R square is: {}'.format(model.score(X_test, y_test)))


# In[795]:


#predicting on the test set
predictions = model.predict(X_test)


# In[796]:


#evaluating the model on mean square error
from sklearn.metrics import mean_squared_error, accuracy_score
print('RMSE is {}'.format(mean_squared_error(y_test, predictions)))


# In[797]:



actual_values = y_test
plt.scatter(predictions, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('Linear Regression Model')
plt.show()


# In[801]:


#Gradient boosting regressor model
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators= 1000, max_depth= 2, learning_rate= .01)
est.fit(X_train, y_train)


# In[807]:


y_train_predict = est.predict(X_train)
y_test_predict = est.predict(X_test)


# In[808]:


est_train = mean_squared_error(y_train, y_train_predict)
print('Mean square error on the Train set is: {}'.format(est_train))


# In[809]:


est_test = mean_squared_error(y_test, y_test_predict)
print('Mean square error on the Test set is: {}'.format(est_test))


# In[810]:


# Random Forrest Regressor model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[811]:


y_pred = regressor.predict(X_test)


# In[815]:


print('RMSE is {}'.format(mean_squared_error(y_test, y_pred)))


# In[ ]:




