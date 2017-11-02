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
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from datetime import datetime
from time import time

from trainClass import trainClass
from models import modelClass
from HoldOutClass import HoldOutClass
import xgbfir
import sys

GPU_FLAG = sys.argv[1]

def ShowHeaderMessage(msg):

    print("\n")
    print("-"*40)
    print("\n      %s" % msg )
    print("\n")
    print("-"*40)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))


    outfile.close()

def MAE(y_true,y_pred):
    mse = metrics.mean_absolute_error(y_true, y_pred)
    #print 'MAE: %.8f' % mse
    return mse

def two_score(y_true,y_pred):
    score = MAE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    return score

def two_scorer():
    return metrics.make_scorer(two_score, greater_is_better=False)
    # change for false if using MSE

def readTestingData(outlier=False):

    print("\n     read properties ......")
    test_df = pd.read_hdf('properties.h5', 'properties')
    print(test_df.shape)
    X_test = test_df.drop(["parcelid"], axis=1)
    X_test.reset_index(drop=True, inplace=True)

    print("-"*40)
    #print("    X_test columns name ....")
    #print(X_test.columns.tolist())

    return X_test

def readTrainingData(outlier=False):

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

    #ShowHeaderMessage("Month counting....")
    #print(X["transaction_month"].value_counts())
    X.reset_index(drop=True, inplace=True)
    #X = X.values

    return X, y_

def interaction3ways(_df):

    df_sel = _df.copy()
    df_sel = df_sel.assign(mlt_finish_taxamt = df_sel.finishedsquarefeet12 * df_sel.taxamount  )
    df_sel = df_sel.assign(mlt_finish_taxdollar = df_sel.finishedsquarefeet12 * df_sel.taxvaluedollarcnt  )
    df_sel = df_sel.assign(mlt_finish_taxamt_taxdollar = df_sel.finishedsquarefeet12 * df_sel.taxamount * df_sel.taxvaluedollarcnt  )

    df_sel = df_sel.assign(div_finish_taxamt =  df_sel.taxamount / df_sel.finishedsquarefeet12 )
    df_sel = df_sel.assign(div_finish_taxdollar = df_sel.taxvaluedollarcnt / df_sel.finishedsquarefeet12 )
    df_sel = df_sel.assign(div_taxamt_taxdollar = df_sel.taxamount /  df_sel.taxvaluedollarcnt  )

    df_sel = df_sel.assign(pow_finish = np.power(df_sel.finishedsquarefeet12,2)  )
    df_sel = df_sel.assign(pow_taxdollar = np.power(df_sel.taxvaluedollarcnt,2)  )
    df_sel = df_sel.assign(pow_taxamt = np.power(df_sel.taxamount,2) )

    del _df
    return df_sel

def blendingDataPreparation(X,y,X_test):

    ghsc1 = ["finishedsquarefeet12","taxamount","taxvaluedollarcnt",
            "transaction_year","transaction_month","transaction_qtr"]

    X_sel = X[ghsc1]
    X_test_sel = X_test[X_sel.columns.tolist()]

    #
    # 3 ways interaction
    #
    X_sel = interaction3ways(X_sel)
    X_test_sel = interaction3ways(X_test_sel)

    missing_col_X = X_sel.columns[X_sel.isnull().any()].tolist()
    missing_col_X_test = X_test_sel.columns[X_test_sel.isnull().any()].tolist()
    ShowHeaderMessage("check missing columns .....")
    print(missing_col_X)
    print(missing_col_X_test)

    ShowHeaderMessage("final shape")
    print(X_sel.shape,X_test_sel.shape,y.shape)

    return X_sel,y,X_test_sel

def modelStart():

    #
    # 1. GradientBoostingRegressor
    #
    g_clfs = [
        GradientBoostingRegressor(n_estimators = 700,
                            max_depth=6,
                            subsample=0.7,
                            learning_rate=0.03,
                            random_state=42)
    ]

    for i in range(2):



def cv_estimate(clf,X,y,X_test):

    # 20% HOLD out
    #X_train,X_val,y_train,y_val = train_test_split(X_sel,y)
    # or another approach

    HoldOutCls = HoldOutClass(X.shape[0],0.2)
    for h in HoldOutCls:
        (train_index,test_index) = h

    X = X.loc[train_index,:]
    y = y[train_index]

    n_folds = 4
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_index,test_index in kf.split(X):

        X_train, X_valid = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_valid = y[train_index], y[test_index]

        start_t = time()
        print(" training ... Fold: %d" % k )
        print( str( datetime.now() )  )
        clf.fit(X_train,y_train)
        print("\n time to train.... %.3f" %  (time() - start_t) )

        clf.predict(X_test)

def blendingDataPreparationXX(X,y,X_test):

    HoldOutCls = HoldOutClass(X.shape[0],0.5)
    for h in HoldOutCls:
        (train_index,test_index) = h

    X = X.loc[test_index,:]
    y = y[test_index]

    X.reset_index(drop=True,inplace=True)
    #y.reset_index(drop=True,inplace=True)

    print("\n    50% resample from training data .... ", X.shape,y.shape)
    print("\n    reindexing after resampling.....")

    ghsc1 = ["finishedsquarefeet12","taxamount","taxvaluedollarcnt",
            "transaction_year","transaction_month","transaction_qtr"]

    X_sel = X[ghsc1]
    X_test_sel = X_test[X_sel.columns.tolist()]
    #
    # 3 ways interaction
    #
    X_sel = interaction3ways(X_sel)
    X_test_sel = interaction3ways(X_test_sel)

    missing_col_X = X_sel.columns[X_sel.isnull().any()].tolist()
    missing_col_X_test = X_test_sel.columns[X_test_sel.isnull().any()].tolist()
    ShowHeaderMessage("check missing columns .....")
    print(missing_col_X)
    print(missing_col_X_test)

    ShowHeaderMessage("final shape")
    print(X_sel.shape,X_test_sel.shape,y.shape)
    #
    #  1. bootstrap , OOB (outofbag)
    #
    n_trees = 30
    n_folds = 5

    clfs = [
        RandomForestRegressor(n_estimators = 500,
                            bootstrap=True,
                            oob_score=True,
                            n_jobs=-1,
                            random_state=42),
        ExtraTreesRegressor(n_estimators = 500,
                            #criterion="mae",
                            bootstrap=True,
                            oob_score=True,
                            n_jobs=-1,
                            random_state=42),
    ]

    #ShowHeaderMessage("ExtraTreesRegressor")
    #for clf in clfs: # ExtraTrees
    #    logoCV_OOB(clf, X_sel,y,X_test_sel)

    g_clfs = [
        GradientBoostingRegressor(n_estimators = 700,
                            max_depth=6,
                            subsample=0.7,
                            learning_rate=0.03,
                            random_state=42)
    ]

    Gradient_OOB(g_clfs[0], X_sel,y,X_test_sel)


def Gradient_OOB(clf, X_sel, y, X_test):

    X_train,X_val,y_train,y_val = train_test_split(X_sel,y)

    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    loss = metrics.mean_absolute_error(y_val, y_pred_val)
    print("Loss: {:.4f}".format(loss))

    acc = clf.score(X_val, y_val)
    print("Accuracy: {:.4f}".format(acc))

    n_estimate_scores = heldout_score(clf, X_val, y_val)
    #print(" val score from staged_decision_function:%.7f" % val_scores)
    x = np.arange(700) + 1

    best_n_estimate_iter = x[ np.argmin(n_estimate_scores) ]
    print("best_iter", best_n_estimate_iter)

    print("length of oob_improvement_ ..", clf.oob_improvement_.shape)
    oob_cumsum = np.cumsum(clf.oob_improvement_) * -1
    best_oob_cumsum_iter = x[ np.argmin(oob_cumsum) ]
    print("best_oob_iter", best_oob_cumsum_iter)

    groups = X_sel["transaction_month"].values
    logo = LeaveOneGroupOut()
    val_scores = np.zeros( (700,),dtype=np.float32  )
    for k, (train, test) in enumerate(logo.split(X_sel, y, groups=groups)):
        #print("%s %s" % (train, test))
        X_train, X_val = X_sel.loc[train],X_sel.loc[test]
        y_train, y_val = y[train],y[test]
        clf.fit(X_train,y_train)
        val_scores += heldout_score(clf, X_val, y_val)

    val_score = val_scores / 700
    best_val_score_iter = x[ np.argmin(val_score) ]
    print("best_val_score_iter", best_val_score_iter)




def logoCV_OOB(clf, X_sel, y, X_test,GB_SW=False):

    #clf.fit(X_sel,y)
    #print("     end of training .....")

    groups = X_sel["transaction_month"].values
    logo = LeaveOneGroupOut()
    k = 0

    randomized_sw = False
    tuned_params = {
                        'n_estimators' : [60,100,200,300,400,500]
    }
    for k, (train, test) in enumerate(logo.split(X_sel, y, groups=groups)):
        #print("%s %s" % (train, test))
        X_train, X_val = X_sel.loc[train],X_sel.loc[test]
        y_train, y_val = y[train],y[test]


        if randomized_sw:
            n_iter_search = 6
            random_search = RandomizedSearchCV(clf,
                            param_distributions=tuned_params,
                            n_iter=n_iter_search,scoring = two_scorer(),
                            error_score=0.01, n_jobs=-1 )
            start = time()
            ShowHeaderMessage("RandomizedSearchCV")
            random_search.fit(X_train, y_train)

            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            #report(random_search.cv_results_)

            print("+")
            print("      Best parameters set found on development set:")
            print(random_search.best_params_)
            params = random_search.get_params()

        clf.fit(X_train,y_train)

        val_scores = .0
        y_pred_val = clf.predict(X_val)

        loss = metrics.mean_absolute_error(y_val,y_pred_val)
        print("\n     MAE : %.7f on Fold:%d" % (loss, k))
        print("     oob score : %.7f" % clf.oob_score_)


def heldout_score(clf, X_test, y_test):

    print("compute deviance scores on X_val and y_val. ")
    score = np.zeros((700,), dtype=np.float32)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        score[i] = clf.loss_(y_test, y_pred)

    return score

def modelRandomizedSearchCV(clf,tuned_params,X_sel,y,X_test):

    groups = X_sel["transaction_month"].values
    logo = LeaveOneGroupOut()
    k = 0
                    #'min_samples_split': [1, 2, 3]}
    for train, test in logo.split(X_sel, y, groups=groups):
        #print("%s %s" % (train, test))
        X_train, X_val = X_sel.loc[train],X_sel.loc[test]
        y_train, y_val = y[train],y[test]
        #print("check dimention...")
        #print(X_train.shape,y_train.shape)
        #print(X_val.shape,y_val.shape)
        n_iter_search = 6
        random_search = RandomizedSearchCV(clf,
                        param_distributions=tuned_params,
                        n_iter=n_iter_search,scoring = two_scorer(),
                        error_score=0.01, n_jobs=-1 )
        start = time()
        ShowHeaderMessage("RandomizedSearchCV")
        random_search.fit(X_train, y_train)

        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        #report(random_search.cv_results_)

        print("+")
        print("      Best parameters set found on development set:")
        print(random_search.best_params_)

        y_pred_val = random_search.predict(X_val)

        loss = MAE(y_val,y_pred_val)
        print("MAE : %7f" % loss)


def valuationProc1(GPU_FLAG):

    GPU_SW = False
    if GPU_FLAG == "GPU":
        GPU_SW = True

    #model_list = ["setXgboostParam1","setXgboostParam2","setXgboostParam3"]
    model_list = ["setXgboostParam3"]
    X, y = readTrainingData(outlier=True)
    X_test = readTestingData()

    print("\n")
    print("     X shape ..........", X.shape, y.shape)
    print("     X_test shape .....", X_test.shape)

    blendingDataPreparation(X,y,X_test)

def writeFile(pred,model_name="xgb"):

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
    output.to_csv('output/stack_{}_{}.csv'.format(model_name,datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

def main():

    valuationProc1(GPU_FLAG)

if __name__ == "__main__":
    main()
