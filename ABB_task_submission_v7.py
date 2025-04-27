#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np # linear algebra
import pandas as pd # data processing
import math
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from xgboost import XGBRegressor
import optuna

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

import pickle

# set seed for reproductibility
np.random.seed(0)


# Load the training data
train_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)

# Group by 'Outlet_Type' and calculate the sum of 'Item_Outlet_Sales'
sales_by_outlet_type = train_df.groupby('Outlet_Type')['Item_Outlet_Sales'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Outlet_Type', y='Item_Outlet_Sales', data=sales_by_outlet_type)
plt.title('Total Item Outlet Sales per Outlet Type')
plt.xlabel('Outlet Type')
plt.ylabel('Total Item Outlet Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# In[2]:


train_df_cat = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df_cat['Item_Fat_Content'] = train_df_cat['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
categorical_features_before_encoding = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size']

target = 'Item_Outlet_Sales'

for feature in categorical_features_before_encoding:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_df_cat, x=feature)
    plt.title(f'Count Plot of {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df_cat, x=feature, y=target)
    plt.title(f'Boxplot of {target} vs. {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.show()



# In[ ]:





# # Supermarket Type 2

# In[3]:


# Supermarket Type 2



# Load the data
train_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df = train_df.loc[train_df['Outlet_Type'].isin(['Supermarket Type2'])].reset_index(drop = True)
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"

test_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df = test_df.loc[test_df['Outlet_Type'].isin(['Supermarket Type2'])].reset_index(drop = True)
test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


original_test_df = test_df.copy() # Keep a copy for submission

target = 'Item_Outlet_Sales'

# --- Helper Functions ---
def impute_item_weight(df):
    item_weight_mean = df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    df['Item_Weight'].fillna(item_weight_mean, inplace=True)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True) # For any remaining NaNs
    return df

def impute_outlet_size(df_train, df_test):
    outlet_size_mode = df_train.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    df_train['Outlet_Size'] = df_train.apply(lambda row: outlet_size_mode[(row['Outlet_Type'], row['Outlet_Location_Type'])] if pd.isnull(row['Outlet_Size']) and (row['Outlet_Type'], row['Outlet_Location_Type']) in outlet_size_mode else row['Outlet_Size'], axis=1)
    df_test['Outlet_Size'] = df_test.apply(lambda row: outlet_size_mode.get((row['Outlet_Type'], row['Outlet_Location_Type']), df_train['Outlet_Size'].mode()[0]) if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1)
    return df_train, df_test

def handle_item_visibility(df):
    df['Item_Visibility'] = np.where(df['Item_Visibility'] == 0, df['Item_Visibility'].median(), df['Item_Visibility'])
    return df

def feature_engineer(df):
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Item_Type_Combined'] = df['Item_Type'].apply(lambda x: x[:4])
    df['Item_Visibility_Ratio'] = df.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x / (x.mean() + 1e-6)) # Avoid division by zero
    return df

def encode_categorical(df_train, df_test):
    le = LabelEncoder()
    df_train['Outlet'] = le.fit_transform(df_train['Outlet_Identifier'])
    df_test['Outlet'] = le.transform(df_test['Outlet_Identifier'])

    categorical_cols = ['Item_Fat_Content', 'Item_Type_Combined', 'Outlet_Location_Type', 'Outlet_Size']
    df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
    return df_train, df_test

def align_columns(df_train, df_test, target):
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    missing_in_test = list(train_cols - test_cols - {target})
    for col in missing_in_test:
        df_test[col] = 0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        df_train[col] = 0

    df_test = df_test[df_train.drop(columns=[target]).columns]
    return df_train, df_test

# --- Data Preprocessing Pipeline ---
train_df = impute_item_weight(train_df)
test_df = impute_item_weight(test_df)

train_df, test_df = impute_outlet_size(train_df, test_df)

train_df = handle_item_visibility(train_df)
test_df = handle_item_visibility(test_df)

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

train_df, test_df = encode_categorical(train_df, test_df)

train_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
test_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train_df, test_df = align_columns(train_df, test_df, target)



# In[4]:


# --- Exploratory Data Analysis (EDA) ---

# 1. Understanding the Target Variable (`Item_Outlet_Sales`)
plt.figure(figsize=(10, 6))
sns.histplot(train_df[target], kde=True)
plt.title('Distribution of Item Outlet Sales')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Frequency')
plt.show()

print(f"Skewness of Item Outlet Sales: {train_df[target].skew():.2f}")


# In[5]:


# 2. Exploring Numerical Features
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Item_Visibility_Ratio']

for feature in numerical_features:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=train_df[feature])
    plt.title(f'Boxplot of {feature}')

    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=train_df[feature], y=train_df[target])
    plt.title(f'Scatter Plot of {feature} vs. {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()


# In[6]:


# 3. Exploring Relationships Between Features - Centered Color Scale

correlation_matrix = train_df[numerical_features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Numerical Features and Target (Centered at 0)')
plt.show()



# In[7]:


# --- Model Selection and Training ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Prepare data for modeling
X = train_df.fillna(0)
X = X.drop(['Item_Outlet_Sales'],axis =1)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()



# In[8]:


y = train_df[target]

corr = pd.concat([y,X],axis =1)
corr = corr.corr()[['Item_Outlet_Sales']]
corr = corr.loc[(corr['Item_Outlet_Sales'] > 0.05) | (corr['Item_Outlet_Sales'] < -0.05)].iloc[1:,:].reset_index(drop = False)
corr['index'].to_list()
corr



# corr


# In[9]:


# Prepare data for modeling
X = train_df[corr['index'].to_list()].fillna(0)
# X = X.drop(['Item_Outlet_Sales'],axis =1)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()


# In[10]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
steps = [
            ('scaler', StandardScaler()),
            ('poly'  , PolynomialFeatures(degree=2)),
            ('model' , Lasso(alpha=30, fit_intercept=True))
       ]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)
ridge_predictions = ridge_pipeline.predict(X_val[corr['index'].to_list()])


ridge_mse  = mean_squared_error(y_val , ridge_predictions)
ridge_rmse = math.sqrt(ridge_mse)
ridge_r2   = r2_score(y_val, ridge_predictions)

print('lasso RMSE  \t         ----> {}'.format(ridge_rmse))
print('lasso R2 Score         ----> {}'.format(ridge_r2))

# print('Training Score  : {}'.format(ridge_pipeline.score(X_train, y_train)))
# print('Test Score      : {}'.format(ridge_pipeline.score(X_test, y_test)))


# In[11]:


# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred_val_rf = rf_model.predict(X_train)
rmse_rf = mean_squared_error(y_train, rf_pred_val_rf, squared=False)
print(f'Random Forest Validation RMSE: {rmse_rf}')
rf_r2   = r2_score(y_train, rf_pred_val_rf)
print('Ridge R2 Score         ----> {}'.format(rf_r2))

# rf_predictions = rf_model.predict(X_test[corr['index'].to_list()])

# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=7)
xgb_model.fit(X_train, y_train)
xgb_pred_val_xgb = xgb_model.predict(X_train)
rmse_xgb = mean_squared_error(y_train, xgb_pred_val_xgb, squared=False)
print(f'XGBoost Validation RMSE: {rmse_xgb}')
xgb_r2   = r2_score(y_train, xgb_pred_val_xgb)
print('Ridge R2 Score         ----> {}'.format(xgb_r2))

# xgb_predictions = xgb_model.predict(X_test[corr['index'].to_list()])

# You can further explore LightGBM and ensemble these models


# In[12]:


rf_predictions = ridge_pipeline.predict(X_test[corr['index'].to_list()])
len(rf_predictions)


# In[13]:


test_df_original = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df_original = test_df_original.loc[test_df_original['Outlet_Type'].isin(['Supermarket Type2'])].reset_index(drop = True)
ridge_predictions_final = ridge_pipeline.predict(X_test[corr['index'].to_list()])
ridge_predictions_final
submission_df = pd.DataFrame({'Item_Identifier': test_df_original['Item_Identifier'], 'Outlet_Identifier': test_df_original['Outlet_Identifier'], 'Item_Outlet_Sales': rf_predictions})
# submission_df.to_csv('submission_ensemble.csv', index=False)
submission_df.to_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_2.csv', index = False)


# # Supermarket Type 3

# In[14]:


# Supermarket Type 3


# Load the data
train_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df = train_df.loc[train_df['Outlet_Type'].isin(['Supermarket Type3'])].reset_index(drop = True)
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


test_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df = test_df.loc[test_df['Outlet_Type'].isin(['Supermarket Type3'])].reset_index(drop = True)
test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


original_test_df = test_df.copy() # Keep a copy for submission

target = 'Item_Outlet_Sales'

# --- Helper Functions ---
def impute_item_weight(df):
    item_weight_mean = df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    df['Item_Weight'].fillna(item_weight_mean, inplace=True)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True) # For any remaining NaNs
    return df

def impute_outlet_size(df_train, df_test):
    outlet_size_mode = df_train.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    df_train['Outlet_Size'] = df_train.apply(lambda row: outlet_size_mode[(row['Outlet_Type'], row['Outlet_Location_Type'])] if pd.isnull(row['Outlet_Size']) and (row['Outlet_Type'], row['Outlet_Location_Type']) in outlet_size_mode else row['Outlet_Size'], axis=1)
    df_test['Outlet_Size'] = df_test.apply(lambda row: outlet_size_mode.get((row['Outlet_Type'], row['Outlet_Location_Type']), df_train['Outlet_Size'].mode()[0]) if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1)
    return df_train, df_test

def handle_item_visibility(df):
    df['Item_Visibility'] = np.where(df['Item_Visibility'] == 0, df['Item_Visibility'].median(), df['Item_Visibility'])
    return df

def feature_engineer(df):
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Item_Type_Combined'] = df['Item_Type'].apply(lambda x: x[:4])
    df['Item_Visibility_Ratio'] = df.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x / (x.mean() + 1e-6)) # Avoid division by zero
    return df

def encode_categorical(df_train, df_test):
    le = LabelEncoder()
    df_train['Outlet'] = le.fit_transform(df_train['Outlet_Identifier'])
    df_test['Outlet'] = le.transform(df_test['Outlet_Identifier'])

    categorical_cols = ['Item_Fat_Content', 'Item_Type_Combined', 'Outlet_Location_Type', 'Outlet_Size']
    df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
    return df_train, df_test

def align_columns(df_train, df_test, target):
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    missing_in_test = list(train_cols - test_cols - {target})
    for col in missing_in_test:
        df_test[col] = 0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        df_train[col] = 0

    df_test = df_test[df_train.drop(columns=[target]).columns]
    return df_train, df_test

# --- Data Preprocessing Pipeline ---
train_df = impute_item_weight(train_df)
test_df = impute_item_weight(test_df)

train_df, test_df = impute_outlet_size(train_df, test_df)

train_df = handle_item_visibility(train_df)
test_df = handle_item_visibility(test_df)

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

train_df, test_df = encode_categorical(train_df, test_df)

train_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
test_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train_df, test_df = align_columns(train_df, test_df, target)

# --- Data Splitting and Scaling ---
X = train_df.drop(columns=[target]).fillna(0)
X = X.drop(columns = ['Outlet_Type'],axis =1)
y = np.log1p(train_df[target]) # Log transform target
X_test = test_df.fillna(0).copy()



# In[15]:


corr = pd.concat([y,X],axis =1)
corr = corr.corr()[['Item_Outlet_Sales']]
corr = corr.loc[(corr['Item_Outlet_Sales'] > 0.05) | (corr['Item_Outlet_Sales'] < -0.05)].iloc[1:,:].reset_index(drop = False)
corr['index'].to_list()
corr



# corr


# In[16]:


# --- Model Selection and Training ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Prepare data for modeling
X = train_df[corr['index'].to_list()].fillna(0)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()



# In[17]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
steps = [
            ('scaler', StandardScaler()),
            ('poly'  , PolynomialFeatures(degree=2)),
            ('model' , Lasso(alpha=1, fit_intercept=True))
       ]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)
ridge_predictions = ridge_pipeline.predict(X_val[corr['index'].to_list()])


ridge_mse  = mean_squared_error(y_val , ridge_predictions)
ridge_rmse = math.sqrt(ridge_mse)
ridge_r2   = r2_score(y_val, ridge_predictions)

print('Ridge RMSE  \t         ----> {}'.format(ridge_rmse))
print('Ridge R2 Score         ----> {}'.format(ridge_r2))

# print('Training Score  : {}'.format(ridge_pipeline.score(X_train, y_train)))
# print('Test Score      : {}'.format(ridge_pipeline.score(X_test, y_test)))


# In[18]:


# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred_val_rf = rf_model.predict(X_train)
rmse_rf = mean_squared_error(y_train, rf_pred_val_rf, squared=False)
print(f'Random Forest Validation RMSE: {rmse_rf}')
rf_r2   = r2_score(y_train, rf_pred_val_rf)
print('Ridge R2 Score         ----> {}'.format(rf_r2))

# rf_predictions = rf_model.predict(X_test[corr['index'].to_list()])

# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=7)
xgb_model.fit(X_train, y_train)
xgb_pred_val_xgb = xgb_model.predict(X_train)
rmse_xgb = mean_squared_error(y_train, xgb_pred_val_xgb, squared=False)
print(f'XGBoost Validation RMSE: {rmse_xgb}')
xgb_r2   = r2_score(y_train, xgb_pred_val_xgb)
print('Ridge R2 Score         ----> {}'.format(xgb_r2))

# xgb_predictions = xgb_model.predict(X_test[corr['index'].to_list()])

# You can further explore LightGBM and ensemble these models


# In[19]:


rf_predictions = ridge_pipeline.predict(X_test[corr['index'].to_list()])
len(rf_predictions)


# In[20]:


rf_predictions


# In[21]:


test_df_original = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df_original = test_df_original.loc[test_df_original['Outlet_Type'].isin(['Supermarket Type3'])].reset_index(drop = True)
ridge_predictions_final = ridge_pipeline.predict(X_test[corr['index'].to_list()])
ridge_predictions_final
submission_df = pd.DataFrame({'Item_Identifier': test_df_original['Item_Identifier'], 'Outlet_Identifier': test_df_original['Outlet_Identifier'], 'Item_Outlet_Sales': ridge_predictions_final})
# submission_df.to_csv('submission_ensemble.csv', index=False)
submission_df.to_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_3.csv', index = False)


# # Supermarket Type 1

# In[22]:


# Supermarket Type 1


# Load the data
train_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df = train_df.loc[train_df['Outlet_Type'].isin(['Supermarket Type1'])].reset_index(drop = True)
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


test_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df = test_df.loc[test_df['Outlet_Type'].isin(['Supermarket Type1'])].reset_index(drop = True)
test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


original_test_df = test_df.copy() # Keep a copy for submission

target = 'Item_Outlet_Sales'

# --- Helper Functions ---
def impute_item_weight(df):
    item_weight_mean = df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    df['Item_Weight'].fillna(item_weight_mean, inplace=True)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True) # For any remaining NaNs
    return df

def impute_outlet_size(df_train, df_test):
    outlet_size_mode = df_train.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    df_train['Outlet_Size'] = df_train.apply(lambda row: outlet_size_mode[(row['Outlet_Type'], row['Outlet_Location_Type'])] if pd.isnull(row['Outlet_Size']) and (row['Outlet_Type'], row['Outlet_Location_Type']) in outlet_size_mode else row['Outlet_Size'], axis=1)
    df_test['Outlet_Size'] = df_test.apply(lambda row: outlet_size_mode.get((row['Outlet_Type'], row['Outlet_Location_Type']), df_train['Outlet_Size'].mode()[0]) if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1)
    return df_train, df_test

def handle_item_visibility(df):
    df['Item_Visibility'] = np.where(df['Item_Visibility'] == 0, df['Item_Visibility'].median(), df['Item_Visibility'])
    return df

def feature_engineer(df):
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Item_Type_Combined'] = df['Item_Type'].apply(lambda x: x[:4])
    df['Item_Visibility_Ratio'] = df.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x / (x.mean() + 1e-6)) # Avoid division by zero
    return df

def encode_categorical(df_train, df_test):
    le = LabelEncoder()
    df_train['Outlet'] = le.fit_transform(df_train['Outlet_Identifier'])
    df_test['Outlet'] = le.transform(df_test['Outlet_Identifier'])

    categorical_cols = ['Item_Fat_Content', 'Item_Type_Combined', 'Outlet_Location_Type', 'Outlet_Size']
    df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
    return df_train, df_test

def align_columns(df_train, df_test, target):
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    missing_in_test = list(train_cols - test_cols - {target})
    for col in missing_in_test:
        df_test[col] = 0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        df_train[col] = 0

    df_test = df_test[df_train.drop(columns=[target]).columns]
    return df_train, df_test

# --- Data Preprocessing Pipeline ---
train_df = impute_item_weight(train_df)
test_df = impute_item_weight(test_df)

train_df, test_df = impute_outlet_size(train_df, test_df)

train_df = handle_item_visibility(train_df)
test_df = handle_item_visibility(test_df)

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

train_df, test_df = encode_categorical(train_df, test_df)

train_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
test_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train_df, test_df = align_columns(train_df, test_df, target)

# --- Data Splitting and Scaling ---
X = train_df.drop(columns=[target]).fillna(0)
X = X.drop(columns = ['Outlet_Type'],axis =1)
y = np.log1p(train_df[target]) # Log transform target
X_test = test_df.fillna(0).copy()



# In[23]:


corr = pd.concat([y,X],axis =1)
corr = corr.corr()[['Item_Outlet_Sales']]
corr = corr.loc[(corr['Item_Outlet_Sales'] > 0.03) | (corr['Item_Outlet_Sales'] < -0.03)].iloc[1:,:].reset_index(drop = False)
corr['index'].to_list()
corr



# corr


# In[24]:


# --- Model Selection and Training ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Prepare data for modeling
X = train_df[corr['index'].to_list()].fillna(0)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()



# In[25]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
steps = [
            ('scaler', StandardScaler()),
            ('poly'  , PolynomialFeatures(degree=2)),
            ('model' , Lasso(alpha=17, fit_intercept=True))
       ]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)
ridge_predictions = ridge_pipeline.predict(X_val[corr['index'].to_list()])


ridge_mse  = mean_squared_error(y_val , ridge_predictions)
ridge_rmse = math.sqrt(ridge_mse)
ridge_r2   = r2_score(y_val, ridge_predictions)

print('Ridge RMSE  \t         ----> {}'.format(ridge_rmse))
print('Ridge R2 Score         ----> {}'.format(ridge_r2))

# print('Training Score  : {}'.format(ridge_pipeline.score(X_train, y_train)))
# print('Test Score      : {}'.format(ridge_pipeline.score(X_test, y_test)))


# In[26]:


# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred_val_rf = rf_model.predict(X_train)
rmse_rf = mean_squared_error(y_train, rf_pred_val_rf, squared=False)
print(f'Random Forest Validation RMSE: {rmse_rf}')
rf_r2   = r2_score(y_train, rf_pred_val_rf)
print('Ridge R2 Score         ----> {}'.format(rf_r2))

# rf_predictions = rf_model.predict(X_test[corr['index'].to_list()])

# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=7)
xgb_model.fit(X_train, y_train)
xgb_pred_val_xgb = xgb_model.predict(X_train)
rmse_xgb = mean_squared_error(y_train, xgb_pred_val_xgb, squared=False)
print(f'XGBoost Validation RMSE: {rmse_xgb}')
xgb_r2   = r2_score(y_train, xgb_pred_val_xgb)
print('Ridge R2 Score         ----> {}'.format(xgb_r2))

# xgb_predictions = xgb_model.predict(X_test[corr['index'].to_list()])

# You can further explore LightGBM and ensemble these models


# In[27]:


rf_predictions = ridge_pipeline.predict(X_test[corr['index'].to_list()])
len(rf_predictions)


# In[28]:


rf_predictions


# In[29]:


test_df_original = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df_original = test_df_original.loc[test_df_original['Outlet_Type'].isin(['Supermarket Type1'])].reset_index(drop = True)
ridge_predictions_final = ridge_pipeline.predict(X_test[corr['index'].to_list()])
ridge_predictions_final
submission_df = pd.DataFrame({'Item_Identifier': test_df_original['Item_Identifier'], 'Outlet_Identifier': test_df_original['Outlet_Identifier'], 'Item_Outlet_Sales': ridge_predictions_final})
# submission_df.to_csv('submission_ensemble.csv', index=False)
submission_df.to_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_1.csv', index = False)


# #  Supermarket Type Grocery Store

# In[30]:


# Supermarket Type Grocery Store



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
train_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\train_v9rqX0R.csv')
train_df = train_df.loc[train_df['Outlet_Type'].isin(['Grocery Store'])].reset_index(drop = True)
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


test_df = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df = test_df.loc[test_df['Outlet_Type'].isin(['Grocery Store'])].reset_index(drop = True)
test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x != 'Regular' else x)
#### Transformation of LF, low fat to Low Fat"


original_test_df = test_df.copy() # Keep a copy for submission

target = 'Item_Outlet_Sales'

# --- Helper Functions ---
def impute_item_weight(df):
    item_weight_mean = df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    df['Item_Weight'].fillna(item_weight_mean, inplace=True)
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True) # For any remaining NaNs
    return df

def impute_outlet_size(df_train, df_test):
    outlet_size_mode = df_train.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    df_train['Outlet_Size'] = df_train.apply(lambda row: outlet_size_mode[(row['Outlet_Type'], row['Outlet_Location_Type'])] if pd.isnull(row['Outlet_Size']) and (row['Outlet_Type'], row['Outlet_Location_Type']) in outlet_size_mode else row['Outlet_Size'], axis=1)
    df_test['Outlet_Size'] = df_test.apply(lambda row: outlet_size_mode.get((row['Outlet_Type'], row['Outlet_Location_Type']), df_train['Outlet_Size'].mode()[0]) if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1)
    return df_train, df_test

def handle_item_visibility(df):
    df['Item_Visibility'] = np.where(df['Item_Visibility'] == 0, df['Item_Visibility'].median(), df['Item_Visibility'])
    return df

def feature_engineer(df):
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Item_Type_Combined'] = df['Item_Type'].apply(lambda x: x[:4])
    df['Item_Visibility_Ratio'] = df.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x / (x.mean() + 1e-6)) # Avoid division by zero
    return df

def encode_categorical(df_train, df_test):
    le = LabelEncoder()
    df_train['Outlet'] = le.fit_transform(df_train['Outlet_Identifier'])
    df_test['Outlet'] = le.transform(df_test['Outlet_Identifier'])

    categorical_cols = ['Item_Fat_Content', 'Item_Type_Combined', 'Outlet_Location_Type', 'Outlet_Size']
    df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)
    return df_train, df_test

def align_columns(df_train, df_test, target):
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    missing_in_test = list(train_cols - test_cols - {target})
    for col in missing_in_test:
        df_test[col] = 0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        df_train[col] = 0

    df_test = df_test[df_train.drop(columns=[target]).columns]
    return df_train, df_test

# --- Data Preprocessing Pipeline ---
train_df = impute_item_weight(train_df)
test_df = impute_item_weight(test_df)

train_df, test_df = impute_outlet_size(train_df, test_df)

train_df = handle_item_visibility(train_df)
test_df = handle_item_visibility(test_df)

train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

train_df, test_df = encode_categorical(train_df, test_df)

train_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
test_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

train_df, test_df = align_columns(train_df, test_df, target)

# --- Data Splitting and Scaling ---
X = train_df.drop(columns=[target]).fillna(0)
X = X.drop(columns = ['Outlet_Type'],axis =1)
y = np.log1p(train_df[target]) # Log transform target
X_test = test_df.fillna(0).copy()



# In[31]:


corr = pd.concat([y,X],axis =1)
corr = corr.corr()[['Item_Outlet_Sales']]
corr = corr.loc[(corr['Item_Outlet_Sales'] > 0.05) | (corr['Item_Outlet_Sales'] < -0.05)].iloc[1:,:].reset_index(drop = False)
corr['index'].to_list()
corr



# corr


# In[32]:


# --- Model Selection and Training ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Prepare data for modeling
X = train_df[corr['index'].to_list()].fillna(0)
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_df.copy()



# In[33]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
steps = [
            ('scaler', StandardScaler()),
            ('poly'  , PolynomialFeatures(degree=1)),
            ('model' , Lasso(alpha=35, fit_intercept=True))
       ]

ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)
ridge_predictions = ridge_pipeline.predict(X_val[corr['index'].to_list()])


ridge_mse  = mean_squared_error(y_val , ridge_predictions)
ridge_rmse = math.sqrt(ridge_mse)
ridge_r2   = r2_score(y_val, ridge_predictions)

print('Ridge RMSE  \t         ----> {}'.format(ridge_rmse))
print('Ridge R2 Score         ----> {}'.format(ridge_r2))

# print('Training Score  : {}'.format(ridge_pipeline.score(X_train, y_train)))
# print('Test Score      : {}'.format(ridge_pipeline.score(X_test, y_test)))


# In[34]:


lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

lin_reg_predictions = lin_reg_model.predict(X_val[corr['index'].to_list()])

lr_mse  = mean_squared_error(y_val , lin_reg_predictions)
lr_rmse = math.sqrt(lr_mse)
lr_r2   = r2_score(y_val, lin_reg_predictions)

print('Ridge RMSE  \t         ----> {}'.format(lr_rmse))
print('Ridge R2 Score         ----> {}'.format(lr_r2))



# In[35]:


# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
rf_model.fit(X_train, y_train)
rf_pred_val_rf = rf_model.predict(X_train)
rmse_rf = mean_squared_error(y_train, rf_pred_val_rf, squared=False)
print(f'Random Forest Validation RMSE: {rmse_rf}')
rf_r2   = r2_score(y_train, rf_pred_val_rf)
print('Ridge R2 Score         ----> {}'.format(rf_r2))

# rf_predictions = rf_model.predict(X_test[corr['index'].to_list()])

# --- XGBoost ---
xgb_model = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=7)
xgb_model.fit(X_train, y_train)
xgb_pred_val_xgb = xgb_model.predict(X_train)
rmse_xgb = mean_squared_error(y_train, xgb_pred_val_xgb, squared=False)
print(f'XGBoost Validation RMSE: {rmse_xgb}')
xgb_r2   = r2_score(y_train, xgb_pred_val_xgb)
print('Ridge R2 Score         ----> {}'.format(xgb_r2))

# xgb_predictions = xgb_model.predict(X_test[corr['index'].to_list()])

# You can further explore LightGBM and ensemble these models


# In[36]:


rf_predictions = ridge_pipeline.predict(X_test[corr['index'].to_list()])
len(rf_predictions)


# In[37]:


test_df_original = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\test_AbJTz2l.csv')
test_df_original = test_df_original.loc[test_df_original['Outlet_Type'].isin(['Grocery Store'])].reset_index(drop = True)
ridge_predictions_final = rf_model.predict(X_test[corr['index'].to_list()])
ridge_predictions_final
submission_df = pd.DataFrame({'Item_Identifier': test_df_original['Item_Identifier'], 'Outlet_Identifier': test_df_original['Outlet_Identifier'], 'Item_Outlet_Sales': rf_predictions})
# submission_df.to_csv('submission_ensemble.csv', index=False)
submission_df.to_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_Grocerry.csv', index = False)


# In[38]:


#### COMBINING ALL 4 ouputs with a merge on the sample Validation Upload file to get the scores


# In[39]:


file_1 = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_1.csv')
file_1


# In[40]:


file_2 = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_2.csv')
file_2


# In[41]:


file_3 = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\v13_3.csv')
file_3


# In[42]:


file_4 = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_Grocerry.csv')
file_4


# In[43]:


concat = pd.concat([file_1,file_2,file_3,file_4],axis =0).reset_index(drop = True)
concat


# In[44]:


formatter = pd.read_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\sample_submission_8RXa3c6.csv')
formatter = formatter.drop(['Item_Outlet_Sales'],axis =1)
formatter = pd.merge(concat,formatter, on =['Item_Identifier','Outlet_Identifier'])

formatter


# In[45]:


formatter.to_csv(r'C:\Users\Rajiv\OneDrive\Documents\Rajiv\Interview Datasets\ABB\Python Outputs\Final_lap_psuh_push\v13_final.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




