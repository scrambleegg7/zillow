#

from trainClass import trainClass
from models import modelClass

import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

from time import time
#import seaborn as #sns

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans

from statsmodels.graphics.gofplots import qqplot_2samples
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor


#import seaborn as sns
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#
# liner regression model
#
from sklearn import linear_model

#
# Aug. 4th,  LightGBM + 2 x XGBoost => 0.0645426
# Stack RidgeModel alpha 0.1

# Aug. 6th  LightGBM + 2 x XGBoost + catbost => 0.065xxx
# SGDRegressor(penalty="elasticnet",loss="huber")
# result much worse ....



# target columns name
target = "logerror"
geoColumns = ["latitude","longitude"]

modelCls = modelClass()

def selectData(df):

    topfeatures = [
        "structuretaxvaluedollarcnt",
        "lotsizesquarefeet",
        "landtaxvaluedollarcnt",
        "longitude",
        "latitude",
        "taxamount",
        "calculatedfinishedsquarefeet",
        "taxvaluedollarcnt",
        "yearbuilt",
        "finishedsquarefeet12",
        "regionidzip",
        "regionidcity",
        "rawcensustractandblock",
        "propertyzoningdesc",
        "censustractandblock",
        "logerror"
    ]
    _df = df[topfeatures].copy()
    return _df

def RMSE(y_true,y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt( mse )
    print 'RMSE: %.6f' % rmse
    return rmse

def MSE(y_true,y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    print 'MSE: %.6f' % mse
    return mse


def two_score(y_true,y_pred):
    score = RMSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    #score = R2(y_true,y_pred)
    return score

def two_scorer():
    return metrics.make_scorer(two_score, greater_is_better=False)
    # change for false if using MSE

def pahse1_gridSearch(x_train, y_train, x_test):

    estimator = lgb.LGBMRegressor(num_leaves=512)

    param_grid = {
        'learning_rate': [0.002, 0.01, 0.05,0.02, 0.03,0.04],
        #'num_leaves': [256,512,1024],
        #'max_depth': [3,4],
        #'boosting_type' : ['gbdt'],
        #'objective' : ['regression'],
        #'metric' : ['l1','l2'],
        'bagging_fraction' : [0.5,0.6,0.7,0.8,0.9],
        #'bagging_freq' : [20,30,40,50],
        'sub_feature' : [0,2,0.3,0.4,0.5],
        #'min_data' : [300,400,500,600,700],
        #'min_hessian' : [0,1,2,5,6,7,9,10],
    }
    gbm = GridSearchCV(estimator, param_grid, cv= 5 , verbose=1)
    gbm.fit(x_train, y_train)
    print('Best parameters found by grid search are:', gbm.best_params_)

    #ngrid=2
    #gbm = RandomizedSearchCV(estimator, param_distributions=param_grid,
    #                                   n_iter=ngrid,cv=5,scoring='roc_auc')
    #gbm.fit(X_train, y_train)

def phase1_run(_x_train, _y_train, _x_test):

    x_train = _x_train.copy()
    x_test = _x_test.copy()

    y_train = _y_train.copy()

    params_list = modelCls.getModelParams()
    params_list = [ v for (k,v) in params_list.items() if "lightgbm" in k ]

    row_num, feat_num = x_train.shape
    _n_splits = 5
    kf = KFold(n_splits=_n_splits, shuffle=True,random_state=42)

    print("   number of folds .... 5")
    print("Sample test shape", x_test.shape)

    base_models_length = len(params_list)
    S_train = np.zeros( (x_train.shape[0],base_models_length)   )
    S_test = np.zeros( (x_test.shape[0], base_models_length) )
    print("S_test shape...",S_test.shape)

    #
    # lgb params_list -> base_models
    #
    for i, params in enumerate(params_list):

        S_test_i = np.zeros( (x_test.shape[0], _n_splits) )

        for j, (train_idx, test_idx) in enumerate( kf.split(x_train) ):
            X_train = x_train[train_idx]
            y_true = y_train[train_idx]
            X_holdout = x_train[test_idx]
            y_holdout = y_train[test_idx]

            d_train = lgb.Dataset(X_train, label=y_true)
            print("\nFitting LightGBM model ... fold:%d" % j)
            clf = lgb.train(params, d_train, 430)
            #clf.fit(X_train, y_train)
            y_pred = clf.predict(X_holdout)

            #showMAE(y_holdout,y_pred)

            S_train[test_idx, i] = y_pred

            S_test_i[:, j] = clf.predict(x_test)

        S_test[:,i] = S_test_i.mean(1)

    print S_train.shape,S_test.shape
    return S_train,S_test

def phase2_xgb_run(_x_train, y_train, _x_test):

    #
    # shold be convert into numpy array
    #
    x_train = _x_train.values
    x_test = _x_test.values
    #y_train = _y_train

    print("+"*30)
    print("\n   after outliers data .... 1% ~ 99%")
    print('   XGBmodel train shape : {}\n  test shape: {}'.format(x_train.shape, x_test.shape))


    params_list = modelCls.setXGBOOSTModel(  np.mean(y_train)  )
    params_list = [ v for (k,v) in params_list.items() ]
    print("\n   length of xgb parameters .. %d" % len(params_list))

    row_num, feat_num = x_train.shape
    _n_splits = 5
    kf = KFold(n_splits=_n_splits, shuffle=True,random_state=42)

    print("\n  xgb : number of folds .... 5")
    print("  Sample test shape", x_test.shape)

    base_models_length = len(params_list)
    S_train = np.zeros( (x_train.shape[0],base_models_length)   )
    S_test = np.zeros( (x_test.shape[0], base_models_length) )
    print("  S_test shape...",S_test.shape)


    for i, xgb_params in enumerate(params_list):

        print("\n   xgb_params ...")
        for (k,v) in xgb_params.items():
            print(k,v)

        S_test_i = np.zeros( (x_test.shape[0], _n_splits) )

        for j, (train_idx, test_idx) in enumerate( kf.split(x_train) ):

            print("\n   fold %d  ...." % j)
            X_TR = x_train[train_idx]
            y_TR = y_train[train_idx]
            print("\n  X_train shape ", X_TR.shape)
            print("\n  y_train shape ", y_TR.shape)

            X_holdout = x_train[test_idx]
            y_holdout = y_train[test_idx]
            print("\n  X_holdout shape ", X_holdout.shape)
            print("\n  y_holdout shape ", y_holdout.shape)


            dtrain = xgb.DMatrix(X_TR, label=y_TR)
            #cv_result = xgb.cv(xgb_params,
            #                   dtrain,
            #                   nfold=5,
            #                   num_boost_round=250,
            #                   early_stopping_rounds=50,
            #                   verbose_eval=10,
            #                   show_stdv=False
            #                  )

            #print("\nFitting XGBOOST model ... fold:%d" % j)

            #num_boost_rounds = len(cv_result)
            num_boost_rounds = 250
            print(" xgb number of boost rounds..",num_boost_rounds)
            # train model
            print("-"*30)
            print(" training .......")
            start_t = time()
            model = xgb.train(dict(xgb_params, silent=1),
                dtrain, num_boost_round=num_boost_rounds)
            print("time to train ....", (time() - start_t))

            dholdout = xgb.DMatrix(X_holdout)
            y_pred = model.predict(dholdout)

            #showMAE(y_holdout,y_pred)
            S_train[test_idx, i] = y_pred


            dtest = xgb.DMatrix(x_test)
            S_test_i[:, j] = model.predict(dtest)

            print("")
        S_test[:,i] = S_test_i.mean(1)

    print S_train.shape,S_test.shape
    return S_train,S_test

def phase2_cat_run(_x_train, y_train, _x_test):

    #
    # shold be convert into numpy array
    #
    x_train = _x_train.values
    x_test = _x_test.values
    #y_train = _y_train

    print("+"*30)
    print("\n   after outliers data .... 1% ~ 99%")
    print('   Cat boost train shape : {}\n  test shape: {}'.format(x_train.shape, x_test.shape))


    #params_list = modelCls.setXGBOOSTModel(  np.mean(y_train)  )
    #params_list = [ v for (k,v) in params_list.items() ]
    #print("\n   length of xgb parameters .. %d" % len(params_list))

    row_num, feat_num = x_train.shape
    _n_splits = 5
    kf = KFold(n_splits=_n_splits, shuffle=True,random_state=42)

    print("\n  catboost : number of folds .... 5")
    print("  Sample test shape", x_test.shape)

    # just only one model
    base_models_length = 1
    S_train = np.zeros( (x_train.shape[0],base_models_length)   )
    S_test = np.zeros( (x_test.shape[0], base_models_length) )
    print("  S_train S_test shape ...",S_test.shape)

    model = CatBoostRegressor(iterations=800,learning_rate=0.005)

    print("    catboost parameters ......")
    print(model.get_params())


    for i in range(1):

        S_test_i = np.zeros( (x_test.shape[0], _n_splits) )

        for j, (train_idx, test_idx) in enumerate( kf.split(x_train) ):

            print("\n   fold %d  ...." % j)
            X_TR = x_train[train_idx]
            y_TR = y_train[train_idx]
            print("\n  X_train shape ", X_TR.shape)
            print("\n  y_train shape ", y_TR.shape)

            X_holdout = x_train[test_idx]
            y_holdout = y_train[test_idx]
            print("\n  X_holdout shape ", X_holdout.shape)
            print("\n  y_holdout shape ", y_holdout.shape)

            # train model
            print("-"*30)
            print(" cat boost training .......")
            start_t = time()
            model.fit(X_TR, y_TR)
            #model = xgb.train(dict(xgb_params, silent=1),
            #    dtrain, num_boost_round=num_boost_rounds)
            print("time to train ....", (time() - start_t))

            #dholdout = xgb.DMatrix(X_holdout)
            y_pred = model.predict(X_holdout)

            #showMAE(y_holdout,y_pred)
            S_train[test_idx, i] = y_pred


            #dtest = xgb.DMatrix(x_test)
            S_test_i[:, j] = model.predict(x_test)

            print("")
        S_test[:,i] = S_test_i.mean(1)

    print S_train.shape,S_test.shape
    return S_train,S_test


def phase2_stacker_run(S_train,S_test,y_true):

    print("-"*30)
    print("  check data shape before running stacker model ")
    print("S_train shape : ",S_train.shape)
    print("S_test shape : ", S_test.shape)
    print("y_true shape : ", y_true.shape)


    print("-"*30)
    print("  Ridge model ....")

    clf = linear_model.Ridge(alpha=0.9)
    #clf = linear_model.SGDRegressor(penalty="elasticnet",loss="huber")
    #clf.fit(X, y)
    clf.fit(S_train,y_true)

    print("coef >>> ",clf.coef_)
    print("intercept >>>", clf.intercept_)
    y_pred = clf.predict(S_train)
    #showMAE(y_true,y_pred)

    y_hat = clf.predict(S_test)

    return y_hat




def model_predict(model,dtest):

    y_hat = model.predict(dtest)
    print("-"*30)
    print(" model prediction ....")
    return y_hat

def showMAE(y_true,y_pred):
    print("y_predict (target) shape ", y_pred.shape)
    residual = y_true - y_pred

    #sns.distplot(residual)
    plt.show()

    print("Mean:%.6f Std:%.6f" % ( np.mean(residual), np.std(residual)) )
    loss = metrics.mean_absolute_error(y_true,y_pred)
    print("MAE : %.7f" % loss)


def showMSE(y_test,y_pred):
    print("y_predict (target) shape ", y_pred.shape)
    residual = y_test - y_pred
    #train_df["residual"] = residual

    loss = metrics.mean_squared_error(y_test,y_pred)
    print("RMSE : %.7f" % np.sqrt( loss ))
    print("MSE : %.7f" % loss)

def showResidualDistribution(y_test,y_pred):
    residual = y_test - y_pred
    #sns.distplot(residual)
    plt.show()

def data_manage_GridSearch():

    trainCls = trainClass(test=True)
    x_train, y_train, x_test = trainCls.makeTrainDataForLightGBM()

    pahse1_gridSearch(x_train, y_train, x_test)

def data_manage_1():

    y_hat = np.zeros(2985217)
    makeSubmitBasedOnLinearRegression(y_hat)

def data_manage_2():

    trainCls = trainClass(test=True)
    x_train, y_train, x_test = trainCls.makeTrainDataForLightGBM()

    TEST_MODEL_NUM = 2
    # number of samples x features = 3
    #S_TRAIN = np.zeros( ( x_train.shape[0],TEST_MODEL_NUM ) )
    #S_TEST = np.zeros( ( x_test.shape[0],TEST_MODEL_NUM )   )
    #
    #  base line model ....
    #
    s_train_light,s_test_light = phase1_run(x_train, y_train, x_test)
    #S_TRAIN[:,0] = s_train[:,0]
    #S_TEST[:,0] = s_test[:,0]

    x_train, y_train, x_test = trainCls.makeTrainDataForXgboost()

    s_train_cat,s_test_cat = phase2_cat_run(x_train, y_train, x_test)
    s_train,s_test = phase2_xgb_run(x_train, y_train, x_test)

    print("-"*30)
    print("LightGBM result shape ....")
    print(s_train_light.shape,s_test_light.shape)
    print("-"*30)
    print("XGBoost result shape ....")
    print(s_train.shape,s_test.shape)
    print("-"*30)
    print("catboost result shape ....")
    print(s_train_cat.shape,s_test_cat.shape)

    #S_TRAIN[:,:] = s_train[:,:]
    #S_TEST[:,:] = s_test[:,:]
    s_train = np.hstack( ( s_train_light,s_train   )    )
    s_test = np.hstack( ( s_test_light,s_test   )    )

    s_train = np.hstack( ( s_train_cat,s_train   )    )
    s_test = np.hstack( ( s_test_cat,s_test   )    )

    print("-"*30)
    print("result Integration - data shape ....")
    print(s_train.shape,s_test.shape)

    y_hat = phase2_stacker_run(s_train,s_test, y_train)

    properties = trainCls.getProp()

    # hold till getting final improved score..
    #makeSubmit(y_hat, properties)
    makeSubmitBasedOnLinearRegression(y_hat)

def makeSubmit2(y_hat,properties):

    y_pred = []
    print("-"*30)
    print("length of y_hat ...", len(y_hat))


def get_features1(df):
    #df.loc[:,"transactiondate"] = pd.to_datetime(df["transactiondate"])
    df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )
    df = df.assign( transactiondate_year = lambda x: x.transactiondate.dt.year   )
    df = df.assign( Month = lambda x: x.transactiondate.dt.month   )
    df = df.assign( transactiondate = lambda x: x.transactiondate.dt.quarter   )
    df = df.fillna(-1.0)
    return df

def makeSubmitBasedOnLinearRegression(y_hat):
    trainCls = trainClass()
    print("")
    print("-"*30)
    print("     re-reading main data ....")
    train = trainCls.readTrain2016()
    properties = trainCls.readProp2016()
    submission = trainCls.makeSampleForTesting()
    print(len(train),len(properties),len(submission))

    def MAE(y, ypred):
        return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

    train = pd.merge(train, properties, how='left', on='parcelid')
    y = train['logerror'].values
    test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
    properties = [] #memory

    exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
    col = [c for c in train.columns if c not in exc]

    train = get_features1(train[col])
    test['transactiondate'] = '2016-01-01' #should use the most common training date
    test = get_features1(test[col])

    reg = linear_model.LinearRegression(n_jobs=-1)
    reg.fit(train, y); print('fit...')
    print(MAE(y, reg.predict(train)))
    train = [];  y = [] #memory

    test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']


    ########################
    ########################
    ##  Combine and Save  ##
    ########################
    ########################

    XGB_WEIGHT = 0.6415
    BASELINE_WEIGHT = 0.0050
    OLS_WEIGHT = 0.0828

    XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

    BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg



    ##### COMBINE PREDICTIONS

    print( "\nCombining XGBoost, LightGBM catboost, and baseline predicitons ..." )
    #lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
    #xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
    baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
    xgb_weight0 = (1 - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)

    #pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test
    pred0 = xgb_weight0*y_hat + baseline_weight0*BASELINE_PRED

    print( "\nCombined XGB/catboost/baseline predictions:" )
    print( pd.DataFrame(pred0).head() )

    print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
    for i in range(len(test_dates)):
        test['transactiondate'] = test_dates[i]
        pred = OLS_WEIGHT*reg.predict(get_features1(test)) + (1-OLS_WEIGHT)*pred0
        submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
        print('predict...', i)

    print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
    print( submission.head() )





def makeSubmit(y_hat,properties):


    y_pred = []
    print("-"*30)
    print("length of y_hat ...", len(y_hat))

    for i,predict in enumerate(y_hat):
        y_pred.append(str(round(predict,4)))
    y_pred=np.array(y_pred)


    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
            '201610': y_pred, '201611': y_pred, '201612': y_pred,
            '201710': y_pred, '201711': y_pred, '201712': y_pred})
    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    from datetime import datetime

    print("  create file ....")
    output.to_csv('sub_stack{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    print("  done ....")



def main():

    # data_manage_GridSearch()
    # data_manage_2()
    data_manage_1()

if __name__ == "__main__":
    main()
