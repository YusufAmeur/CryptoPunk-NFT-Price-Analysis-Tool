#!/usr/bin/env python
# coding: utf-8

# # **Import Packages and Data**

# Import Packages

# In[ ]:



import csv

import numpy as np
import pandas as pd # dataframes & data analysis!

from sklearn.model_selection import train_test_split  # module to split our data into train and test sets
import numpy
from sklearn.linear_model import LogisticRegression as LogReg
import statsmodels.api as sm # package to build the linear regression model

from sklearn.preprocessing import StandardScaler #for scaling the data
from sklearn.model_selection import train_test_split # module to split our data into train and test sets
import seaborn as sns #Library utilised for Visualisations
import matplotlib.pyplot as plt #Library utilised for Visualisations
from sklearn import metrics
from time import time
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestRegressor # Utilised for Random Forest Regression
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV # Number of trees in random forest

import plotly.express as px #Library utilised for Visualisations
import matplotlib.pyplot as plt #Library utilised for Visualisations


# Import Data

# In[ ]:


##Here the dataset is read into a DataFrame.

df = pd.read_json("txn_history-2021-10-07.jsonl", lines=True)
len(df)


# In[ ]:


##What does the data look like?
df.head()


# # **Exploratory Data Analysis**

# In[ ]:


##Check the dataset to ensure everything is successfully imported, including all columns.

pd.set_option('max_columns', None)
df.head(1000)


# In[ ]:


## Review of how many rows and columns is in the dataset.
df.shape


# In[ ]:


## Review of the datatypes, notice there are a lot of objects.
df.dtypes


# In[ ]:


##What are the features of the data
df.describe()


# What does each column actually depict? Here we are exploring each unique value from all DataSeries

# In[ ]:


##What are the columns of the dataset
df.columns


# In[ ]:


## What values are found in the 'txn_type' column?

df.txn_type.unique()


# In[ ]:


## What values are found in the 'to' column?
df.to.unique()


# In[ ]:


## What values are found in the 'date' column?
df.date.unique()


# In[ ]:


## What values are found in the 'timestamp' column?
df.timestamp.unique()


# In[ ]:


## What values are found in the 'source' column?
df.source.unique()


# In[ ]:


## What values are found in the 'eth' column?
df.eth.unique()


# In[ ]:


## What values are found in the 'punk_id' column?
df.punk_id.unique()


# In[ ]:


## What values are found in the 'from_wallet_address' column?
df.from_wallet_address.unique()


# In[ ]:


## What values are found in the 'to_wallet_address' column?
df.to_wallet_address.unique()


# In[ ]:


## What values are found in the 'type' column?
df.type.unique()


# In[ ]:


## What values are found in the 'accessories' column?
df.accessories.unique()


# ------------------------------------------------------------
# 

# Upon review of unique values of each column, it is evident there is a lot of repeatability across columns. Hence, 'from_wallet_address', 'to_wallet_address', 'timestamp', 'from' and 'to' were dropped.

# In[ ]:


## Columns "from_wallet_address, to_wallet_address, timestamp, from and to" have been dropped, due to not providing any meaningful information that will add value to the tool.

try:
    df.drop(columns = ['from_wallet_address', 'to_wallet_address', 'timestamp', 'from', 'to'], inplace = True)
except:
    print("Already dropped")

df.head()


# **Are there Nulls within the Data?**

# In[ ]:


## Function to identify nulls within the whole dataset, highlighting the percentage said nulls take up within the whole Dataset.

def null_vals(dataframe):
    null_vals = dataframe.isnull().sum()
    total_cnt = len(dataframe)
    null_vals = pd.DataFrame(null_vals, columns=['null'])
    null_vals['percent'] = round((null_vals['null'] / total_cnt) * 100, 3)
    return null_vals.sort_values('percent', ascending=False)


null_vals(df)


# In[ ]:


##There are numerous nulls within the eth DataSeries, lets take a closer look
df[df['eth'].isnull()]


# There are numerous nulls within the 'eth' DataSeries. Such nulls represent the minting of the CryptoPunk (which required little/no eth), the transfer of a CryptoPunk from one wallet to another, or an open-ended offer being made on a CryptoPunk, which was subsequently either ignored or rejected.
# 
# Considering the predictive model of the tool needing representative prices each CryptoPunk was sold for, the dataframe needs to be filtered solely on Sales. While one can make a debate that 'Offers' can depict potential value of a product, some offers can be extremely anomalous and skew the data. Hence, was excluded from the dataset.
# 

# In[ ]:


#Create a variable solely for CryptoPunk sales
CryptoPunk_Sales = df[df["txn_type"] == "Sold"]
CryptoPunk_Sales


# In[ ]:


## Function to identify nulls within the refined CryptoPunk_Sales dataset.

def null_vals(dataframe):
    null_vals = dataframe.isnull().sum()
    total_cnt = len(dataframe)
    null_vals = pd.DataFrame(null_vals, columns=['null'])
    null_vals['percent'] = round((null_vals['null'] / total_cnt) * 100, 3)
    return null_vals.sort_values('percent', ascending=False)


null_vals(CryptoPunk_Sales)


# All Nulls were successfully removed.

# ---------------

# **Analysis of Duplicates, are they present?**

# DataSeries 'type' and 'accessories' prove to be an obstacle in cleaning (due to being in the form of a list). Hence, we sought to temporarily drop such columns to check if duplicates are present.

# In[ ]:


#Temporary removal of 'type', 'accessories' DataSeries
CryptoPunk_Sales1= CryptoPunk_Sales.drop(columns = ['type', 'accessories']).copy()
CryptoPunk_Sales1.shape


# In[ ]:


# Function to locate the exact position of duplicated data. Notice the result is a float, suggesting some data was duplicated more than once.

g = CryptoPunk_Sales1.duplicated(subset=None, keep=False).reset_index()
count = 0
number = 0
for i in g[0]:
    count += 1
    if i == True:
        number += 1
        print(i, "Position of duplicate:", count - 1)
print("Total Number of duplicates is", number / 2)


# In[ ]:


# Reduction in the size of the data confirms a successful removal of duplicates.

CryptoPunk_Sales1.drop_duplicates(keep='last', inplace = True)
CryptoPunk_Sales1.shape


# Considering DataSeries 'type' and 'accessories' provide core information for the model, such data needs to be returned.
# 

# In[ ]:


#Joining of duplicate-free data to original dataset
CryptoPunk_Sales = CryptoPunk_Sales1.join(CryptoPunk_Sales,lsuffix='',rsuffix='_right')
CryptoPunk_Sales


# In[ ]:


## Duplicated columns need to be removed.
# Notice that all txn_type are all sold, thus would be ideal to remove from the dataset

CryptoPunk_Sales = CryptoPunk_Sales.drop(columns = ['txn_type', 'txn_type_right', 'date_right', 'source_right', 'eth_right', 'punk_id_right', 'source'])
CryptoPunk_Sales


# ---------------------

# **Unpacking of DataSeries containing lists.**

# In[ ]:


# Unpacking of the 'type' DataSeries
CryptoPunk_Sales = CryptoPunk_Sales.explode("type")
CryptoPunk_Sales


# In[ ]:


# Unpacking of the 'accessory' DataSeries. Considering that the current accessories-list is still needed, unpacking of the DataSeries is assigned to an external variable.

Accessories = CryptoPunk_Sales['accessories'].explode()
Accessories


# Considering the sheer number of accessories associated with all CryptoPunks and the limited size of the dataset, consideration into the curse of dimensionality needs to be taken. While its ideal to use all accessories as potential feature columns for the model, to prevent over-fitting of the model, an alternative method of depicting attribute rarity is taken.
# 
# The lower quartile range of the 'accessories' DataSeries is used, representing lowest occurring accessories. Such accessories are denoted by a numerical value depending on how many ‘rare’ accessories are present.

# In[ ]:


## Accessories have been individually unpacked, counted and divided by the total number of accessories present across all CryptoPunk assets.

Accessories_Clean = Accessories.value_counts(normalize= True).reset_index().sort_values('accessories',ascending=False)
Accessories_Clean['accessories'] = Accessories_Clean['accessories'].round(4) * 100

Accessories_Clean.rename(columns={"accessories":"Occurrence_of_Accessory (%)", "index": "Accessory"}, inplace=True)  ## renaming my columns!
Accessories_Clean


# In[ ]:


## The lower quartile range is 0.460000 within the data, and will be the threshold of an accessory being 'rare'
Accessories_Clean.describe()


# In[ ]:


#All Accessories considered to be 'rare' have been assigned to a list, utilised to act as a dictionary of rare traits.

Rare_Accessories = Accessories_Clean[Accessories_Clean["Occurrence_of_Accessory (%)"] < 0.460000].reset_index()
rare_list = list(Rare_Accessories['Accessory'])
rare_list


# In[ ]:


#Function which assigns a numerical value to the number of rare accessories found within each CryptoPunk Sale.
def rarity_checker(accessory_list):
    count = 0
    for accessory in accessory_list:
        if accessory in rare_list:
            count += 1
    return count

CryptoPunk_Sales['rarity_check'] = CryptoPunk_Sales['accessories'].apply(lambda accessory_list: rarity_checker(accessory_list))
CryptoPunk_Sales


# --------------------------

# **Visualisations of the Data**

# What CryptoPunks tend to be more expensive?

# In[ ]:


##Figure depicting Max sold price of a CryptoPunk relative to type
fig = px.bar(CryptoPunk_Sales.groupby("type").agg({"eth": "max"}).sort_values(by="eth").reset_index('type'),
             x="type", y="eth", color="type", title="CryptoPunk Max Sold Price by Type")
fig.show()


# In[ ]:


##Fluctuations of transactions per day

dates = df['date'].value_counts().sort_index().rename_axis('date').reset_index(name='counts')
plt.figure(figsize=(20,10))
plt.bar(dates['date'], dates['counts'], label="All Transactions")
plt.legend()
plt.xticks(rotation=60)
plt.ylim(0, 1000)
plt.title("Transactions per Day")
plt.ylabel("Number of Transactions")
plt.xlabel("Date")
plt.show()


# -----------------------

# NFT fluctuations in price are dependent on the cryptomarket as a whole. While there are many factors are involved in dictating the price of both the NFT and Cryptocurrency market, date is of the most influential factors.
# 
# Here, the date was categorised annually to prevent the curse of dimensionality and subsequent over-fitting of data.

# In[ ]:


def year_check(date):
    year = date.year
    years_since = 2021 - year
    return years_since


# In[ ]:


CryptoPunk_Sales['year'] = CryptoPunk_Sales['date'].apply(lambda x: year_check(x))


# In[ ]:


##Here we dropped accessories, date and punk_id due to not adding any statistical value from this point forward
CryptoPunk_Sales = CryptoPunk_Sales.drop(columns = ['accessories', 'date', 'punk_id'])
CryptoPunk_Sales


# All the data has been cleaned, processed and for the model.

# # **Feature Engineering**

# In[ ]:


# Compiling the DataSeries for the train sample data, and removing the 'eth' column as this will be the y value

feature_train = list(CryptoPunk_Sales.columns)
feature_train.remove('eth')

feature_train


# In[ ]:


# Defining the X and y values

X = CryptoPunk_Sales[feature_train]
y = CryptoPunk_Sales['eth']


# In[ ]:


# Creating 4 portions of data which will be used for fitting and predicting values.

# X_train includes all our columns under 'CryptoPunk_Sales' for our train model, we also specified the test size as '0.2' meaning 80% of our complete data will be used to create the model whilst the other 20%, X_test, will be set aside and used after to 'test' the model
# y_train and y_test is the dependent variable, in our case 'Eth',
# We used the random state 45 across all our models to ensure validity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


# In[ ]:


# Feature engineering
# Then one hot encoded the columns: 'type' and 'year' as well as adding a constant

def feature_eng(inputed):
    inputed = pd.get_dummies(inputed, columns = ['type', 'year'], prefix = ['type', 'year'], drop_first = True)
    inputed = sm.add_constant(inputed)
    return inputed


# In[ ]:


# Transforming the train data to be feature-engineered

X_train = feature_eng(X_train)


# In[ ]:


# Transforming the test data to be feature-engineered
X_test = feature_eng(X_test)


# ----------------------

# # **Linear Regression Model Creation**

# In[ ]:


# Creating and fitting a linear regression model using the feature engineered data

lin_reg = sm.OLS(y_train, X_train)
results = lin_reg.fit()
print(results.summary())


# In[ ]:


# Creating a prediction based on the linear regression model created earlier and extracting the RMSE for the train data

X_train['y_pred'] = results.predict(X_train)
rmse = sm.tools.eval_measures.rmse(y_train, X_train['y_pred'])
print(rmse)


# In[ ]:


lin_reg = sm.OLS(y_test, X_test)
test_results = lin_reg.fit()
print(test_results.summary())


# In[ ]:


# Lastly, we are predicting using the X_test set and extracting the RMSE score


X_test['y_pred'] = test_results.predict(X_test)
rmse = sm.tools.eval_measures.rmse(y_test, X_test['y_pred'])
print(rmse)


# An RMSE value of 59.7 (Train) and 51.4 (Test) were achieved. But can this be improved with another model?

# --------------------------------------------------------------------------------------

# # **Decision Tree Regression**

# In[ ]:


## Function that determines that optimal hyper-parameters to use within the Random Forest Regressor.

n_estimators = [int(x) for x in np.linspace(start = 2, stop = 40, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Random grid represents the best hyper-parameters

random_grid


# In[ ]:


## Fit the data to the model, which utilised the optimal hyper-parameters

#First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)


# In[ ]:


## Check to see what are the best hyper-parameters relative to the X_train dataset <- whats the difference between this and randomgrid
rf_random.best_params_


# In[ ]:


##Here we can make a prediction based on the model.
rf_pred = rf_random.best_estimator_.predict(X_train)


# In[ ]:


##Here a rmse evaluation was made on the train data
rmse = sm.tools.eval_measures.rmse(y_train, rf_pred)
print(rmse)


# In[ ]:


##Here we can make a prediction based on the model.
test_rf_pred = rf_random.best_estimator_.predict(X_test)


# In[ ]:


##Here a rmse evaluation was made on the test data
test_rmse = sm.tools.eval_measures.rmse(y_test, test_rf_pred)
print(test_rmse)


# Utilisation of Decision Tree Regression seemed to not improve our RMSE score. Thus, the model that utilises Linear Regression should be prioritised.

# ---------------------------------

# # **Predictive tool**

# In[ ]:


##Creation of an external dataframe, prepared for input

input_prediction = pd.DataFrame(columns=['rarity_check','type_Female','type_Ape','type_Male','type_Zombie','year_1','year_2','year_3','year_4'])
input_prediction


# In[ ]:


##input-functions prepared for implementation of CryptoPunk features

Type_of_punk= input("What is the type of your CryptoPunk?: ")

Predict_Accessories = []
Predict_Accessories.append(input("Enter list of Accessories: "))

Year= int(input("Enter Year in which CryptoPunk was purchased: "))


# In[ ]:


##Function to input data into input_prediction Dataframe based off input

def predictive_type_check(type):
    if type == 'Male':
        input_prediction['type_Female'] = 0
        input_prediction['type_Ape'] = 0
        input_prediction['type_Male'] = 1
        input_prediction['type_Zombie'] = 0

    elif type == "Female":
        input_prediction['type_Female'] = 1
        input_prediction['type_Ape'] = 0
        input_prediction['type_Male'] = 0
        input_prediction['type_Zombie'] = 0

    elif type == "Ape":
        input_prediction['type_Female'] = 0
        input_prediction['type_Ape'] = 1
        input_prediction['type_Male'] = 0
        input_prediction['type_Zombie'] = 0

    elif type == "Alien":
        input_prediction['type_Female'] = 0
        input_prediction['type_Ape'] = 0
        input_prediction['type_Male'] = 0
        input_prediction['type_Zombie'] = 0

    elif type == "Zombie":
        input_prediction['type_Female'] = 0
        input_prediction['type_Ape'] = 0
        input_prediction['type_Male'] = 0
        input_prediction['type_Zombie'] = 1

    else:
        print('Not a Valid CryptoPunk Type')

predictive_type_check(Type_of_punk)


# In[ ]:


#Function which assigns a numerical value to the number of rare accessories found within each Accessory input.
def Predictive_rarity_checker(Predict_Accessories):
    count = 0
    for accessory in Predict_Accessories:
        if accessory in rare_list:
            count += 1
    return count

b = Predictive_rarity_checker(Predict_Accessories)
input_prediction['rarity_check'] = b


# In[ ]:


####Function to input data into input_prediction dataframe based off input

def predictive_year_check(Year):
    years_since = 2021 - Year
    return years_since

c = predictive_year_check(Year)

if c == 1:
    input_prediction['year_1'] = 1
    input_prediction['year_2'] = 0
    input_prediction['year_3'] = 0
    input_prediction['year_4'] = 0

elif c == 2:
    input_prediction['year_1'] = 0
    input_prediction['year_2'] = 1
    input_prediction['year_3'] = 0
    input_prediction['year_4'] = 0

elif c == 3:
    input_prediction['year_1'] = 0
    input_prediction['year_2'] = 0
    input_prediction['year_3'] = 1
    input_prediction['year_4'] = 0

elif c == 4:
    input_prediction['year_1'] = 0
    input_prediction['year_2'] = 0
    input_prediction['year_3'] = 0
    input_prediction['year_4'] = 1


# In[ ]:


##loop that reflects what data within the CryptoPunk data reflects the attributes input

for items in X_test:
    if X_test['rarity_check'] == input_prediction['rarity_check'] and X_test['type_Female'] == input_prediction['type_Female'] and X_test['type_Ape'] == input_prediction['type_Ape'] and X_test['type_Male'] == input_prediction['type_Male'] and X_test['type_Zombie'] == input_prediction['type_Zombie'] and X_test['year_1'] == input_prediction['year_1'] and X_test['year_2'] == input_prediction['year_2'] and X_test['year_3'] == input_prediction['year_3'] and X_test['year_4'] == input_prediction['year_4']: print(f'The predicted ETH value of such CryptoPunk is:', X_test['y_pred'][1])

