# A little memory heavy but should generate some options for anyone stuck
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np; np.random.seed(17)
from datetime import timedelta
import random; random.seed(17)
import pandas as pd
import numpy as np
from sklearn import *
from multiprocessing import *
import datetime as dt
import gc; gc.enable()
#from catboost import *
#import xgboost as xgb
#import lightgbm as lgb

cal = USFederalHolidayCalendar()
holidays = [d.date() for d in cal.holidays(start='2016-01-01', end='2017-12-31').to_pydatetime()]
business = [d.date() for d in pd.date_range('2016-01-01', '2017-12-31') if d not in pd.bdate_range('2016-01-01', '2017-12-31')]
holidays_prev = [d + timedelta(days=-1) for d in holidays]
holidays_after = [d + timedelta(days=1) for d in holidays]

#Train 16
train = pd.read_csv('../input/train_2016_v2.csv')[:10000] #limiting for kernel
train = pd.merge(train, pd.read_csv('../input/properties_2016.csv'), how='left', on='parcelid')
train_17 = pd.read_csv('../input/train_2017.csv')[:10000] #limiting for kernel
train_17 = pd.merge(train_17, pd.read_csv('../input/properties_2017.csv'), how='left', on='parcelid')
train = pd.concat((train, train_17), axis=0, ignore_index=True).reset_index(drop=True)
del train_17; gc.collect();

ecol = [c for c in train.columns if train[c].dtype == 'object'] + ['taxdelinquencyflag','propertycountylandusecode','propertyzoningdesc','parcelid','ParcelId','logerror','transactiondate']
col = [c for c in train.columns if c not in ['taxdelinquencyflag','propertycountylandusecode','propertyzoningdesc','parcelid','ParcelId','logerror','transactiondate']]
dcol = col.copy()
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
one_hot = {c: list(train[c].unique()) for c in col}

#df_dd_sheets = [pd.read_excel('../input/zillow_data_dictionary.xlsx', sheetname=i) for i in range(8)]
#print(df_dd_sheets[0].head())

def transform_df(df):
    try:
        df = pd.DataFrame(df)
        df['null_vals'] = df.isnull().sum(axis=1)
        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        df['transactiondate_year'] = df['transactiondate'].dt.year
        df['transactiondate_month'] = df['transactiondate'].dt.month
        df['transactiondate_day'] = df['transactiondate'].dt.day
        df['transactiondate_dow'] = df['transactiondate'].dt.dayofweek
        df['transactiondate_wd'] = df['transactiondate'].dt.weekday
        df['transactiondate_h'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays else 0)
        df['transactiondate_hp'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays_prev else 0)
        df['transactiondate_ha'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays_after else 0)
        df['transactiondate_b'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in business else 0)
        df['transactiondate_quarter'] = df['transactiondate'].dt.quarter
        df = df.fillna(-1.0)
        for c in dcol:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
        for c in one_hot:
            if len(one_hot[c])>2 and len(one_hot[c]) < 10:
                for val in one_hot[c]:
                    df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    except Exception as e:
        print(e)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def MAE(y, pred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-pred[i]) for i in range(len(y))]) / len(y)

train = multi_transform(train)
col = [c for c in train.columns if c not in ecol]

print('LinearRegressor...')
#temp fix to Sklearn MLK error by @Yakolle Zhang
reg = linear_model.Lasso()
reg.fit(train[col], train['logerror'])

reg = linear_model.LinearRegression(n_jobs=-1)
reg.fit(train[col], train['logerror'])
print(MAE(train['logerror'], reg.predict(train[col])))
del train;  gc.collect();

#Pred 16
test = pd.read_csv('../input/sample_submission.csv')[:10000] #limiting for kernel
test_col = [c for c in test.columns]
test = pd.merge(test, pd.read_csv('../input/properties_2016.csv'), how='left', left_on='ParcelId', right_on='parcelid')
test_dates = ['2016-10-01','2016-11-01','2016-12-01']
test_columns = ['201610','201611','201612']
for i in range(len(test_dates)):
    transactiondate =  dt.date(*(int(s) for s in test_dates[i].split('-')))
    dr = pd.date_range(transactiondate, transactiondate + timedelta(days=27))
    test['transactiondate'] = np.random.choice(dr, len(test['ParcelId']))
    test = multi_transform(test) #keep order
    test[test_columns[i]] = reg.predict(test[col])
    print('predict...', test_dates[i])

#Pred 17
test = test[test_col]
test = pd.merge(test, pd.read_csv('../input/properties_2017.csv'), how='left', left_on='ParcelId', right_on='parcelid')
test_dates = ['2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201710','201711','201712']
for i in range(len(test_dates)):
    transactiondate =  dt.date(*(int(s) for s in test_dates[i].split('-')))
    dr = pd.date_range(transactiondate, transactiondate + timedelta(days=27))
    test['transactiondate'] = np.random.choice(dr, len(test['ParcelId']))
    test = multi_transform(test) #keep order
    test[test_columns[i]] = reg.predict(test[col])
    print('predict...', test_dates[i])

test = test[test_col]
test.to_csv('submission.csv.gz', index=False, compression='gzip', float_format='%.4f')
