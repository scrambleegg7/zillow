# Parameters
#XGB_WEIGHT = 0.6400
#BASELINE_WEIGHT = 0.0050
#OLS_WEIGHT = 0.0856
#OLS_WEIGHT = 0.0828


#XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

#BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


# just added another parameters

# Parameters
XGB_WEIGHT = 0.6400
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


#
# Your submission scored 0.0649909, which is not an improvement of your best score. Keep trying!
# Aug. 15th
#


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt


##### READ IN RAW DATA

#print( "\nReading data from disk ...")
#prop = pd.read_csv('../input/properties_2016.csv')
#train = pd.read_csv("../input/train_2016_v2.csv")

from trainClass import trainClass

##### READ IN RAW DATA

print( "\nReading data from disk ...")
#prop = pd.read_csv('../input/properties_2016.csv')
#train = pd.read_csv("../input/train_2016_v2.csv")

trainCls = trainClass()
train = trainCls.readTrain2016()
prop = trainCls.readProp2016()




################
################
##  LightGBM  ##
################
################

# This section is (I think) originally derived from SIDHARTH's script:
#   https://www.kaggle.com/sidharthkumar/trying-lightgbm
# which was forked and tuned by Yuqing Xue:
#   https://www.kaggle.com/yuqingxue/lightgbm-85-97
# and updated by me (Andy Harless):
#   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
# and a lot of additional changes have happened since then,
#   the most recent of which are documented in my comments above


##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
#x_train['Ratio_1'] = x_train['taxvaluedollarcnt']/x_train['taxamount']
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    print(c)
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)



##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
#clf = lgb.train(params, d_train, 430)

clf = lgb.train(params, d_train, 1000)

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")

#sample = pd.read_csv('../input/sample_submission.csv')
sample = trainCls.readSampleSub()


print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(prop, on='parcelid', how='left')
print("   ...")
del sample; gc.collect()
print("   ...")
#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']
x_test = df_test[train_columns]

print("-"*30)
print("X_TEST columns name .....")
print(x_test.columns)
print("-"*30)

print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
pred = clf.predict(x_test)

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(pred).head() )


y_pred=[]
for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('output/sub_lightGBM_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
