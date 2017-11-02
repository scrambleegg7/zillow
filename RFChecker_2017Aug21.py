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
from sklearn.model_selection import KFold

from datetime import datetime
from time import time

from trainClass import trainClass
from models import modelClass
import xgbfir

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))


    outfile.close()


def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

def readTestingData(outlier=False):

    # original
    #trainCls = trainClass()
    #orig_train = trainCls.getTrain()
    #y_orig = orig_train["logerror"].values.astype(np.float32)

    print("\n     read properties ......")
    test_df = pd.read_hdf('properties.h5', 'properties')
    print(test_df.shape)
    X_test = test_df.drop(["parcelid"], axis=1)
    X_test.reset_index(drop=True, inplace=True)

    print("-"*40)
    print("    X_test columns name ....")
    #print(X_test.columns.tolist())

    return X_test

def readTrainingData(outlier=False):

    # original
    #trainCls = trainClass()
    #orig_train = trainCls.getTrain()
    #y_orig = orig_train["logerror"].values.astype(np.float32)

    print("\n     read train data ......")

    train_df = pd.read_hdf('storage.h5', 'train')

    if outlier:
        train_df=train_df[ train_df.logerror > -0.4 ]
        train_df=train_df[ train_df.logerror < 0.419 ]

    print(train_df.shape)

    y_ = train_df["logerror"].values.astype(np.float32)

    #print("\n")
    #print("-"*30)
    #print("check original data with reading h5 data.....")
    #print("MAE .. %.7f" % MAE(y_orig,y_))

    X = train_df.drop(["parcelid","logerror"], axis=1)
    training_columns_list = X.columns.tolist()

    print("-"*40)
    print("    X_test columns name ....")
    #print(training_columns_list)
    X.reset_index(drop=True, inplace=True)
    #X = X.values

    return X, y_

def rf_valuation(mfunc_name,X,y,X_test):

    model = modelClass()
    training_columns_list = X.columns.tolist()
    X_test = X_test[training_columns_list]

    rf_model = getattr(model,mfunc_name)()

    n_folds = 2
    kf = KFold(n_splits=n_folds,random_state=42)
    S_test = np.zeros( (X_test.shape[0],1) )
    S_train = np.zeros( (X.shape[0],1) )
    S_test_i = np.zeros((X_test.shape[0], n_folds))

    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_valid = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_valid = y[train_index], y[test_index]

        start_t = time()
        print("-"*40)
        print("\n      start RandomForest training ... Fold:%d %s" % (k, str( datetime.now() )) )

        rf_model.fit(X_train,y_train)
        print("\n       time to randomForest train.... %.3f" %  (time() - start_t) )

        y_pred = rf_model.predict(X_valid)
        loss = MAE(y_valid,y_pred)
        print("\n")
        print("     ** loss --> %.7f" % loss)

        y_test = rf_model.predict(X_test)
        S_test_i[:,k] = y_test

    S_test[:,0] = S_test_i.mean(1)

    return S_test

def lr_valuation(mfunc_name,X,y,X_test):

    model = modelClass()

    training_columns_list = X.columns.tolist()
    X_test = X_test[training_columns_list]

    lr_model = getattr(model,mfunc_name)()

    n_folds = 2
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    S_test = np.zeros( (X_test.shape[0],1) )
    S_train = np.zeros( (X.shape[0],1) )
    S_test_i = np.zeros((X_test.shape[0], n_folds))

    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_valid = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_valid = y[train_index], y[test_index]

        start_t = time()
        print("-"*40)
        print("\n      start LinearRegression training ... Fold:%d %s" % (k, str( datetime.now() )) )
        lr_model.fit(X_train,y_train)

        print("\n      time to train.... %.3f" %  (time() - start_t) )
        print("\n      Predicting with LinearRegression ...")
        y_pred = lr_model.predict(X_valid)
        loss = MAE(y_valid,y_pred)

        S_train[test_index, 0] = y_pred

        print("\n")
        print("\n      ** LinearRegression loss --> %.7f" % loss)

        y_test = lr_model.predict(X_test)
        S_test_i[:,k] = y_test

    S_test[:,0] = S_test_i.mean(1)
    print("S_test shape", S_test.shape)
    print("S_train shape", S_train.shape)

    return S_test

def lrModel():

    model_list = ["linearRegressionModel"]
    X, y = readTrainingData(outlier=True)
    X_test = readTestingData()

    print("\n")
    print("     X shape ..........", X.shape, y.shape)
    print("     X_test shape .....", X_test.shape)

    for m in model_list:
        lr_pred = lr_valuation(m,X,y,X_test)
        writeFile(lr_pred,"lr")

def rfModel():

    model_list = ["randomForestModel"]
    X, y = readTrainingData(outlier=True)
    X_test = readTestingData()

    print("\n")
    print("     X shape ..........", X.shape, y.shape)
    print("     X_test shape .....", X_test.shape)

    for m in model_list:
        rf_pred = rf_valuation(m,X,y,X_test)
        writeFile(rf_pred,"rf")

def writeFile(pred,model_name="liner"):

    properties = pd.read_hdf('properties.h5', 'properties')

    y_pred = []
    for i,predict in enumerate( pred.ravel() ):
        y_pred.append(str(round(predict,4)))
    y_pred=np.array(y_pred)

    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
            '201610': y_pred, '201611': y_pred, '201612': y_pred,
            '201710': y_pred, '201711': y_pred, '201712': y_pred})
    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    output.to_csv('output/bl_{}_{}.csv'.format( model_name, datetime.now().strftime('%Y%m%d_%H%M%S') ) , index=False)

def main():

    lrModel()
    rfModel()

    #xgbModel_loop()
    #valuation()
    #xgb_valuation(model)

if __name__ == "__main__":
    main()
