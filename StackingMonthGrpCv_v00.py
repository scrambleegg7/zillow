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
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold

from datetime import datetime
from time import time

from trainClass import trainClass
from models import modelClass
from HoldOutClass import HoldOutClass
import xgbfir
import sys

import argparse


#GPU_FLAG = sys.argv[1]

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

def FeaturesSelection1(X,y,X_test):

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
    print("check missing columns .....")
    print(missing_col_X)
    print(missing_col_X_test)

    print("final shape")
    print(X_sel.shape,X_test_sel.shape,y.shape)

    return X_sel,y,X_test_sel

def FeaturesSelection2(X,y,X_test):

    ghsc1 = ["finishedsquarefeet12","regionidcity","structuretaxvaluedollarcnt",
            "transaction_year","transaction_month","transaction_qtr"]

    X_sel = X[ghsc1]
    X_test_sel = X_test[X_sel.columns.tolist()]

    missing_col_X = X_sel.columns[X_sel.isnull().any()].tolist()
    missing_col_X_test = X_test_sel.columns[X_test_sel.isnull().any()].tolist()
    print("check missing columns .....")
    print(missing_col_X)
    print(missing_col_X_test)

    print("final shape")
    print(X_sel.shape,X_test_sel.shape,y.shape)

    return X_sel,y,X_test_sel

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))

    outfile.close()

def MAE(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

def readTestingData(log_sw=False):

    # original
    #trainCls = trainClass()
    #orig_train = trainCls.getTrain()
    #y_orig = orig_train["logerror"].values.astype(np.float32)

    print("\n     read properties ......")

    if log_sw:
        test_df = pd.read_hdf('properties_log.h5', 'properties')
    else:
        print("     read normalized Data....")
        test_df = pd.read_hdf('properties_norm.h5', 'properties')

    print(test_df.shape)
    X_test = test_df.drop(["parcelid"], axis=1)
    X_test.reset_index(drop=True, inplace=True)

    print("-"*40)
    print("    X_test columns name ....")
    #print(X_test.columns.tolist())

    return X_test

def readTrainingData(outlier=False,log_sw=False):

    # original
    #trainCls = trainClass()
    #orig_train = trainCls.getTrain()
    #y_orig = orig_train["logerror"].values.astype(np.float32)

    print("\n     read train data ......")
    if log_sw:
        train_df = pd.read_hdf('storage_log.h5', 'train')
    else:
        train_df = pd.read_hdf('storage_norm.h5', 'train')

    if outlier:
        train_df=train_df[ train_df.logerror > -0.4 ]
        train_df=train_df[ train_df.logerror < 0.419 ]

    train_df.reset_index(drop=True,inplace=True)

    print(train_df.shape)

    y_ = train_df["logerror"].values.astype(np.float32)

    #print("\n")
    #print("-"*30)
    #print("check original data with reading h5 data.....")
    #print("MAE .. %.7f" % MAE(y_orig,y_))

    X = train_df.drop(["parcelid","logerror","abs_logerror"], axis=1)
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

    #rf_model = model.randomForestModel()

    n_folds = 2
    kf = KFold(n_splits=n_folds,random_state=42)
    S_test = np.zeros( (X_test.shape[0],1) )
    S_train = np.zeros( (X.shape[0],1) )
    S_test_i = np.zeros((X_test.shape[0], n_folds))

    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_valid = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_valid = y[train_index], y[test_index]

        start_t = time()
        print("Random Forest - training ... Fold: %d" % k )
        print( str( datetime.now() )  )

        rf_model.fit(X_train,y_train)
        print("\n time to train.... %.3f" %  (time() - start_t) )

        y_pred = rf_model.predict(X_valid)
        loss = MAE(y_valid,y_pred)
        print("\n")
        print("     ** loss --> %.7f" % loss)

        y_test = rf_model.predict(X_test)
        S_test_i[:,k] = y_test

    S_test[:,0] = S_test_i.mean(1)

    return S_test

def xgb_select_features(mfunc_name,X,y,X_test,GPU=False):

    #
    # set HoldOutClass to get 50% test data ...
    #
    HoldOutCls = HoldOutClass(X.shape[0],0.5)
    for h in HoldOutCls:
        (train_index,test_index) = h

    X = X.loc[test_index,:]

    model = modelClass()

    ghsc1 = ["finishedsquarefeet12","taxamount","taxvaluedollarcnt"]

    mask = X.transaction_month < 10
    X_sel = X[mask,ghsc1]

    training_columns_list = X.columns.tolist()
    X_test = X_test[training_columns_list]

    xgb_params = getattr(model,mfunc_name)()
    plst = xgb_params[0]


def test_valuation(m,bst_model, X_test=None, m1X_frame=None):

    #
    #
    # X_test valuation
    # 1. get model
    # 2. valuation with features used in training ..
    # 3. return result by date
    # 4. save full X_test matrix for next M1 model (Stacking)
    # 5. in 2nd round, attach M1 feature on X_test DataFrame
    # 6. valuation with M1 training model
    # 7. return result by date

    # save CSV file ....

    trainCls = trainClass()
    submission = trainCls.readSampleSub()


    test_dates = ['2016-10-01','2016-11-01',
                    '2016-12-01','2017-10-01',
                    '2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']

    X_frame = np.zeros( (X_test.shape[0],len(test_dates)) )
    for i,test_date in enumerate(test_dates):
        mydt = pd.to_datetime(test_date)
        X_test["transaction_year"] = mydt.year
        X_test["transaction_month"] = mydt.month
        X_test["transaction_qtr"] = mydt.quarter

        if "M1" in m:
            X_test["M1"] = m1X_frame[:,i]

        dtest = xgb.DMatrix(X_test)

        pred = bst_model.predict(dtest,bst_model.best_ntree_limit)
        X_frame[:,i] = pred.ravel()

        month_col = str(mydt.year) +str(mydt.month)
        submission[month_col] = [float(format(x, '.4f')) for x in pred]
        print('predict...', i)

    print( "\nWriting results to disk ..." )
    submission.to_csv('output/stack_model/{}_{}.csv'.format(m,datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

    return X_frame

def xgb_valuation_holdout(mfunc_name,X_train,y_train,X_holdout,y_holdout,X_test,GPU=False):

    model = modelClass()

    training_columns_list = X_train.columns.tolist()
    _X_test = X_test[training_columns_list]

    xgb_params = getattr(model,mfunc_name)()
    plst = xgb_params[0]
    num_boost_rounds = xgb_params[1]
    print(plst)
    print("     initial boost rounds..", num_boost_rounds)

    params = {}
    for (k,v) in plst:
        params[k] = v

    #
    # using holdout
    #
    xgtrain = xgb.DMatrix(X_train, y_train)
    dtrain = xgb.DMatrix(X_train)
    xgval = xgb.DMatrix(X_holdout, y_holdout)
    dholdout = xgb.DMatrix(X_holdout)
    print("     original feature ....", X_train.shape,X_holdout.shape)
    #dtest = xgb.DMatrix(X_test)
    watchlist  = [ (xgtrain,'train'),(xgval,'eval')]
    bst = xgb.train(plst,
                    xgtrain,
                    num_boost_rounds,
                    watchlist,
                    verbose_eval=200,
                    early_stopping_rounds=50)

    #
    # add y prediction as new feature (X)
    #
    y_M1_pred = bst.predict(dtrain,ntree_limit=bst.best_ntree_limit)
    X_train["M1"] = y_M1_pred.ravel().astype(np.float32)

    #
    # make test model
    #
    m1X_frame = test_valuation(mfunc_name,bst,_X_test)

    #
    # add y prediction as new feature (X_holdout)
    #
    y_M1_pred_holdout = bst.predict(dholdout,ntree_limit=bst.best_ntree_limit)
    X_holdout["M1"] = y_M1_pred_holdout.ravel().astype(np.float32)

    xgtrain = xgb.DMatrix(X_train, y_train)
    xgval = xgb.DMatrix(X_holdout, y_holdout)
    print("     ** add new feature from prediction result ...", X_train.shape,X_holdout.shape)

    watchlist  = [ (xgtrain,'train'),(xgval,'eval')]
    bst = xgb.train(plst,
                    xgtrain,
                    num_boost_rounds,
                    watchlist,
                    verbose_eval=200,
                    early_stopping_rounds=50)

    #
    # If early stopping is enabled during training,
    # you can get predictions from the best iteration
    # with bst.best_ntree_limit:
    #
    dholdout = xgb.DMatrix(X_holdout)
    y_pred_holdout = bst.predict(dholdout,ntree_limit=bst.best_ntree_limit)

    print(  MAE(y_holdout,y_pred_holdout))
    print('best ite:',bst.best_iteration)
    print('best score:',bst.best_score)

    #
    # make m1 test model
    #
    xframe_ = test_valuation(mfunc_name+"_M1",bst,_X_test,m1X_frame)

def xgb_valuation(mfunc_name,X_train,y_train,X_test,GPU=False):

    model = modelClass()

    training_columns_list = X_train.columns.tolist()
    X_test = X_test[training_columns_list]

    xgb_params = getattr(model,mfunc_name)()
    plst = xgb_params[0]

    #
    # GPU
    #
    if GPU:
        params = {}
        params['gpu_id'] = 0
        params['max_bin'] = 16
        params['tree_method'] = 'gpu_exact'
        plst.extend( list(params.items()) )

    num_boost_rounds = xgb_params[1]

    print(plst)
    print("     initial boost rounds..", num_boost_rounds)

    params = {}
    for (k,v) in plst:
        params[k] = v
    # cross-validation
    #print( "Running XGBoost CV ..." )
    # cross-validation
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_test)
    cv_result = xgb.cv(params,
                       dtrain,
                       nfold=4,
                       num_boost_round=1000,
                       early_stopping_rounds=50,
                       verbose_eval=100,
                       show_stdv=False
                      )
    num_boost_rounds = len(cv_result)
    print("     appropriate boost rounds...",num_boost_rounds)
    #print(cv_result[num_boost_rounds-1])
    #
    #
    bst = xgb.train(dict(params, silent=1), dtrain,
            num_boost_round=num_boost_rounds)
    #
    create_feature_map(X_train.columns.tolist())
    filename = "zillow_norm_%s.xlsx" % (mfunc_name)
    xgbfir.saveXgbFI(bst, feature_names=X_train.columns.tolist(), OutputXlsxFile = filename)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df_sort = df.sort_values("fscore", ascending=False   )
    print(df_sort.loc[:10,:])

    return num_boost_rounds


def CrossValidationFullTrain(mfunc_name,X,y,X_test):

    model = modelClass()

    training_columns_list = X.columns.tolist()
    _X_test = X_test[training_columns_list]

    xgb_params = getattr(model,mfunc_name)()
    plst = xgb_params[0]
    num_boost_rounds = xgb_params[1]
    print(plst)
    print("     initial boost rounds..", num_boost_rounds)

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True)

    residuals = np.zeros( X.shape[0] )

    #
    # 1. KFold -->
    for k, (train_index,test_index) in enumerate( kf.split(X) ):
        X_train, X_val = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_val = y[train_index], y[test_index]

        #
        # 2. Group by Month CV   eg. train : 1 2 3 4 5 6 7 8 9 10 11 ==> valid : 12
        #




        xgtrain = xgb.DMatrix(X_train,y_train)
        xgval = xgb.DMatrix(X_val,y_val)
        watchlist  = [ (xgtrain,'train'),(xgval,'eval')]
        y_mean = np.mean(y_train)
        base_sc_param = {'base_score':y_mean}

        plst.extend( list( base_sc_param.items() ) )
        bst = xgb.train(plst,
                        xgtrain,
                        num_boost_rounds,
                        watchlist,
                        verbose_eval=400,
                        early_stopping_rounds=50)

        dval = xgb.DMatrix(X_val)
        y_pred_val = bst.predict(dval,ntree_limit=bst.best_ntree_limit)
        #loss = MAE(y_val,y_pred_val)
        #y_preds.append( loss)
        residuals[test_index] = (y_val - y_pred_val)
        print("     cross-validation : %d" % int(k+1))
        #print("     MAE:%.7f" % loss)

    create_feature_map(X_train.columns.tolist())
    filename = "zillow_norm_%s.xlsx" % (mfunc_name+"_cv")
    xgbfir.saveXgbFI(bst, feature_names=X_train.columns.tolist(), OutputXlsxFile = filename)

    #'output/fig/{}_{}.csv'.format(m,), index=False)
    dstr = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_filename = "output/fig/%s_%s" % (mfunc_name+"_cv_resid",dstr)
    plt.scatter(range(X.shape[0]),residuals,c='r')
    plt.savefig(fig_filename)
    #print("     Average MAE score %.7f" % np.mean(loss))

def xgbModel_loop(X,y,X_test,GPU_SW=False):

    #model_list = ["xModel1","xModel4","xModel5","xModel6","xModel7",
    #                "xModel8","xModel9","xModel10"]
    #model_list = ["xModel1","xModel4","xModel5"]
    model_list = ["setXgboostParam1","setXgboostParam2","setXgboostParam3"]


    #
    # 1. Holdout model
    #
    #for m in model_list:
    #    print("-"*30)
    #    print("     model_name: %s " % m)

    #    print("\n")
    #    print("     get num_boost_rounds ....")
    #    for i in range(2):
    #        X_train,X_holdout,y_train,y_holdout = HoldoutData_split(X,y,int(i*10))
    #        num_boost_rounds = xgb_valuation_holdout(m,X_train,y_train,X_holdout,y_holdout,X_test,GPU_SW)

    #
    # 2. Full train model ( 5 Kfolds train and valuation ..)
    #    cross-validation approach
    for m in model_list:
        print("-"*30)
        print("     model_name: %s " % m)
        CrossValidationFullTrain(m,X,y,X_test)


        #writeFile(xgb_pred)

def rfModel():

    model_list = ["randomForestModel"]
    X, y = readTrainingData(outlier=True)
    X_test = readTestingData()

    print("\n")
    print("     X shape ..........", X.shape, y.shape)
    print("     X_test shape .....", X_test.shape)

    for m in model_list:
        rf_pred = rf_valuation(m,X,y,X_test)
        writeFile(rf_pred)

def writeFile(pred):

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
    output.to_csv('output/stack_xgb_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

def readArg():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logconv", help="log data ..",
                        action="store_true")
    parser.add_argument("-g", "--gpu", help="GPU swith for Xgboost",
                        action="store_true")
    parser.add_argument("-s", "--select", help="features selection for Xgboost",
                        action="store_true")
    args = parser.parse_args()

    log_sw = False
    GPU_SW = False
    feat_sel_sw = False

    if args.logconv:
        print("\n")
        print("    log data is read ..  ")
        log_sw = True
    else:
        print("\n")
        print("    ** normalized data is read ** ..  ")

    if args.gpu:
        print("\n")
        print("    ** GPU processor for XGBoost ** ..  ")
        GPU_SW = True

    if args.select:
        print("\n")
        print("    ** Feature selection for XGBoost ** ..  ")
        feat_sel_sw = True


    return log_sw,GPU_SW, feat_sel_sw

def main():

    log_sw,GPU_SW, feat_sel_sw = readArg()

    X_test = readTestingData(log_sw)
    X,y = readTrainingData(outlier=True,log_sw=log_sw)
    #rfModel()
    #
    # feature selection
    #
    if feat_sel_sw:
        X,y,X_test = FeaturesSelection1(X,y,X_test)

    xgbModel_loop(X,y,X_test,GPU_SW)

if __name__ == "__main__":
    main()
