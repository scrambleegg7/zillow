# Parameters
# XGB_WEIGHT = 0.6840

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


train = pd.read_hdf('storage.h5', 'train')
train=train[ train.logerror > -0.4 ]
train=train[ train.logerror < 0.419 ]

properties = pd.read_hdf('properities.h5', 'properities')
print(properties.shape)
x_test = properties.drop(["parcelid"], axis=1)
x_test.reset_index(drop=True, inplace=True)


x_train=train.drop(['parcelid', 'logerror'], axis=1)
print("--    training columns ....")
print(len(x_train.columns))
training_columns_list = x_train.columns.tolist()

x_test = x_test[training_columns_list]

y_train = train["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))


print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 4,  # 5
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    #'base_score': 0.0,
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
num_boost_rounds = 300

print("\nXGBoost tuned with CV in:")
print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
print("num_boost_rounds="+str(num_boost_rounds))


# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
ceate_feature_map(x_train.columns.tolist())
xgbfir.saveXgbFI(model, feature_names=x_train.columns.tolist(), OutputXlsxFile = 'zillow_ohe.xlsx')

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
