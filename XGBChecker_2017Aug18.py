# Parameters
# XGB_WEIGHT = 0.6840
XGB_WEIGHT = 0.63
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0550

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
import operator

from sklearn import metrics

from datetime import datetime


from trainClass import trainClass

import xgbfir

##### READ IN RAW DATA

print( "\nReading data from disk ...")
#prop = pd.read_csv('../input/properties_2016.csv')
#train = pd.read_csv("../input/train_2016_v2.csv")

trainCls = trainClass()
train = trainCls.getTrain()
properties = trainCls.getProp()



# Aug. 16th, 2017
# Your submission scored 0.0648888, which is not an improvement of your best score. Keep trying!
# Loss( MAE ) -->  0.049257
# LB : 0.0648888

# Aug. 17th, 2017
# after data feature integration .....
# ##  https://www.kaggle.com/jeru666/zillow-revamped-with-memory-reduction

# Loss (MAE) --> 0.049027
# LB : 0.0648580
# max_depth : 5

# Aug. 17th, 2017
# after data feature integration .....
# Loss (MAE) --> 0.031274
# max_depth : 10

# Aug. 18th, 2017
# after data feature integration .....
# Loss( MAE ) -->  0.050609
# max_depth : 6
# use log for every fields ...

# drop fields
# parcelid','propertyzoningdesc','propertycountylandusecode','fireplaceflag'


def safe_log(x):
    try:
        l = np.log(x)
    except ZeroDivisionError:
        l = .0
    except RunTimeWarning:
        l = .0
    return l

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))


    outfile.close()

################
################
##  XGBoost   ##
################
################

# This section is (I think) originally derived from Infinite Wing's script:
#   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
# inspired by this thread:
#   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# but the code has gone through a lot of changes since then



##### PROCESS DATA FOR XGBOOST
##
print("-"*30)
print("-- fill out missing value with something value but 0 ...")

print( "\nProcessing object data for XGBoost ...")
for c in properties.columns:
    #properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        print("Columns ..... %s" % c)
        print(" change to np.int8 and fill with 10e-6..")
        properties[c].fillna(0, inplace=True)
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))
        #properties[c] = properties[c].astype(np.float32)
        #print( properties[c].value_counts() )


##
##  Aug. 16, 2017
##  based on below advices .....
##  https://www.kaggle.com/jeru666/zillow-revamped-with-memory-reduction
##




#properties['hashottuborspa'].fillna(0, inplace=True)
#properties['fireplaceflag'].fillna(0, inplace=True)
#properties['taxdelinquencyflag'].fillna(0, inplace=True)

#properties.hashottuborspa = properties.hashottuborspa.astype(np.int8)
#properties.fireplaceflag = properties.fireplaceflag.astype(np.int8)
#properties['taxdelinquencyflag'].replace( 'Y', 1, inplace=True)
#properties.taxdelinquencyflag = properties.taxdelinquencyflag.astype(np.int8)


properties['yearbuilt'] = properties['yearbuilt'].fillna(2016)
#--- Dropping the 'transactiondate' column now ---
#properties = properties.drop('transactiondate', 1)
properties['unitcnt'].fillna(properties['unitcnt'].median(), inplace=True)

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

diffs = {}
for i in properties.columns[properties.isnull().any()].tolist():
    print("+"*30)
    print("max-min : %s" % i)
    print( properties[i].max()  - properties[i].min()  )
    diff = properties[i].max() - properties[i].min()
    diffs[i] = diff

diff_df = pd.DataFrame(  {"col_name":diffs.keys(),"diff_cols":diffs.values()   }  )
diff_df = diff_df[ diff_df.diff_cols > 100. ]
print( diff_df["col_name"].values.tolist() )
print( "total number : ", diff_df.shape[0] )

for col in diff_df["col_name"].values.tolist():
    print("convert to log(x) ... ", col)
    properties[col] = properties[col].fillna( properties[col].median()  )
    properties.loc[:,col] = safe_log( properties[col].values )


missing_col = properties.columns[properties.isnull().any()].tolist()
print(missing_col)
for col in missing_col:
    print("%s fill with zero ....",col)
    properties[col].fillna( 0,inplace=True )
print('There are {} missing columns'.format(len(missing_col)))

print("-"*30)
print( "\nchecking object data for XGBoost ...")
for col in properties.columns:
    if properties[col].dtype != object:
        if properties[col].dtype == float:
            print("%s changed float32.." % col)
            properties[col] = properties[col].astype(np.float32)







train = train.merge(properties, how='left', on='parcelid')



#x_train = train.drop(['parcelid', 'logerror','transactiondate'], axis=1)
print("-- propertycountylandusecode propertyzoningdesc --> dropped..")
x_test = properties.drop(['parcelid','propertyzoningdesc','propertycountylandusecode','fireplaceflag'], axis=1)

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
             'propertyzoningdesc','propertycountylandusecode','fireplaceflag'], axis=1)
print("--    training columns ....")
print(len(x_train.columns))
print( x_train.columns.tolist() )

y_train = train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))




##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 6,  # 5
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cross-validation
#print( "Running XGBoost CV ..." )
#cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   nfold=5,
#                   num_boost_round=350,
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )
#num_boost_rounds = len(cv_result)
num_boost_rounds = 250

print("\nXGBoost tuned with CV in:")
print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
print("num_boost_rounds="+str(num_boost_rounds))


# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
ceate_feature_map(x_train.columns.tolist())
xgbfir.saveXgbFI(model, feature_names=x_train.columns.tolist(), OutputXlsxFile = 'zillowFI.xlsx')

importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

print(df.sort_values("fscore", ascending=False   )  )




dtrain_test = xgb.DMatrix(x_train)
y_hat = model.predict(dtrain_test)
loss = metrics.mean_absolute_error(y_train,y_hat)
print("\nLoss( MAE ) -->  %.6f" % loss)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )




y_pred=[]
for i,predict in enumerate(xgb_pred1):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv('output/sub_XGBM_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
