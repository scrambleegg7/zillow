
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from sklearn import metrics
##### READ IN RAW DATA

from sklearn.model_selection import train_test_split

from trainClass import trainClass

print( "\nReading data from disk ...")
trainCls = trainClass()
train = trainCls.getTrain()
properties = trainCls.getProp()

##### PROCESS DATA FOR XGBOOST


##
print("-"*30)
print("-- fill out missing value with something value ...")

print("-- hashottuborspa fireplaceflag taxdelinquencyflag --> 0 ")

##
##  Aug. 16, 2017
##  based on below advices .....
##  https://www.kaggle.com/jeru666/zillow-revamped-with-memory-reduction
##

properties['hashottuborspa'] = properties['hashottuborspa'].fillna(0)
properties['fireplaceflag'] = properties['fireplaceflag'].fillna(0)
properties['taxdelinquencyflag'] = properties['taxdelinquencyflag'].fillna(0)

properties.hashottuborspa = properties.hashottuborspa.astype(np.int8)
properties.fireplaceflag = properties.fireplaceflag.astype(np.int8)
properties['taxdelinquencyflag'].replace( 'Y', 1, inplace=True)
properties.taxdelinquencyflag = properties.taxdelinquencyflag.astype(np.int8)


properties['yearbuilt'] = properties['yearbuilt'].fillna(2016)
#--- Dropping the 'transactiondate' column now ---
#properties = properties.drop('transactiondate', 1)

properties['unitcnt'] = properties['unitcnt'].fillna(properties['unitcnt'].mode()[0])

#
#
# replace with Random value
#
#  regionidcity
#
#

min_ = properties["regionidcity"].min()
max_ = properties["regionidcity"].max()
mask = properties["regionidcity"].isnull()
dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidcity"].values )  ) ))
properties.loc[mask, "regionidcity"] = dfrand[mask].values

#
#  regionidneighborhood
#
min_ = properties["regionidneighborhood"].min()
max_ = properties["regionidneighborhood"].max()
mask = properties["regionidneighborhood"].isnull()
dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidneighborhood"].values )  ) ))
properties.loc[mask, "regionidneighborhood"] = dfrand[mask].values

#
#  regionidzip
#
min_ = properties["regionidzip"].min()
max_ = properties["regionidzip"].max()
mask = properties["regionidzip"].isnull()
dfrand = pd.DataFrame(data = np.random.randint(min_,max_, (len(properties["regionidzip"].values )  ) ))
properties.loc[mask, "regionidzip"] = dfrand[mask].values

#
#

properties["latitude"] = properties["latitude"] * 10e6
properties["longitude"] = properties["longitude"] * 10e6

for i in properties.columns[properties.isnull().any()].tolist():
    properties[i] = properties[i].fillna(0)

missing_col = properties.columns[properties.isnull().any()].tolist()
print(missing_col)
print('There are {} missing columns'.format(len(missing_col)))




print("-"*30)
print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
#    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
#        lbl = LabelEncoder()
        print(c)
#        lbl.fit(list(properties[c].values))
#        properties[c] = lbl.transform(list(properties[c].values))

train = train.merge(properties, how='left', on='parcelid')



#x_train = train.drop(['parcelid', 'logerror','transactiondate'], axis=1)
print("-- propertycountylandusecode propertyzoningdesc --> dropped..")
x_test = properties.drop(['parcelid','propertyzoningdesc','propertycountylandusecode'], axis=1)

x_test["transactiondate"] = "2016-10-01"
x_test['transactiondate'] = pd.to_datetime(x_test['transactiondate'])
x_test['transaction_month'] = x_test.transactiondate.dt.month.astype(np.int64)
x_test['transaction_day'] = x_test.transactiondate.dt.weekday.astype(np.int64)
x_test = x_test.drop(['transactiondate'], axis=1)

print("   x_test columns .....")
print( len(x_test.columns) )
print( x_test.columns.tolist() )

# shape
#print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers

# senario 1 --> select outlier with -0.4 to 0.419

train=train[ train.logerror > -0.4 ]
train=train[ train.logerror < 0.419 ]

# scenario 2 -->



train['transactiondate'] = pd.to_datetime(train['transactiondate'])
#--- Creating three additional columns each for the month and day ---
train['transaction_month'] = train.transactiondate.dt.month.astype(np.int64)
train['transaction_day'] = train.transactiondate.dt.weekday.astype(np.int64)


x_train=train.drop(['parcelid', 'logerror','transactiondate',
             'propertyzoningdesc','propertycountylandusecode'], axis=1)
print("--    training columns ....")
print(len(x_train.columns))
print( x_train.columns.tolist() )


y_train = train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)


# shape
print("")
print("-"*40)
print('   Shape train after implementation ..: {}\n   Shape test after implementation ..: {}'.format(x_train.shape, x_test.shape))
print('   \n Shape y (target) ...{}'.format( y_train.shape ))


print("+"*30)
print("     RUN CATBOOST by Russia .....  ")

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed':42,
    'depth':5,
    #'use_best_model': True
}

#clf = CatBoostRegressor(iterations=3000,learning_rate=0.03, random_seed=42, depth=5)
clf = CatBoostRegressor(**cat_params)


#X_train, X_validation, y_train_, y_validation = train_test_split(x_train,
#            y_train, train_size=0.7, random_state=1234)


print(clf.get_params())
clf.fit(x_train, y_train)

features_ = clf.get_feature_importance(x_train,y_train)
df_feat_cat = pd.DataFrame({"column_name":x_train.columns.tolist(),"weights":features_} )
print(  df_feat_cat.sort_values("weights",ascending=False) )

y_pred = clf.predict(x_train)

loss = metrics.mean_absolute_error(y_train,y_pred)
print("-"*30)
print("    catboost base MAE score... %.7f" %  loss)
#print( "\nXGBoost predictions:" )
#print( pd.DataFrame(xgb_pred).head() )
