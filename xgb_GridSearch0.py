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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans

from statsmodels.graphics.gofplots import qqplot_2samples
from sklearn.model_selection import KFold

from scipy import stats


import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

#from catboost import CatBoostClassifier
#from catboost import CatBoostRegressor


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

def readModelClass(y_mean=0.0):


    params_list = modelCls.setXGBOOSTModel(  y_mean  )
    xgb_params_list = [ v for (k,v) in params_list.items() ]
    print("\n   length of xgb parameters .. %d" % len(xgb_params_list))

    return xgb_params_list

def GridSearch_Valuation(_x_train, y_train, _x_test, xgb_params, GridSearchSW=False):

    x_train = _x_train.values
    x_test = _x_test.values


    print("+"*30)
    print("          XGB model parameters to be used ...")
    for (k,v) in xgb_params.items():
        print(k,v)

    row_num, feat_num = x_train.shape
    _n_splits = 5
    kf = KFold(n_splits=_n_splits, shuffle=True,random_state=42)

    print("\n  xgb : number of folds .... 5")
    print("  Sample test shape", x_test.shape)

    test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']

    S_test_i = np.zeros( (_n_splits,  x_test.shape[0], len(test_columns)) )
    print("  S_test_i shape (summary on date ..)...",S_test_i.shape)

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


        if not GridSearchSW:
            dtrain = xgb.DMatrix(X_TR, label=y_TR)
            print("\nFitting Xgb model ... ")

            num_boost_rounds = 1000
            print(" xgb number of boost rounds..",num_boost_rounds)
            # train model
            print("-"*30)
            print(" training .......")
            start_t = time()
            model = xgb.train(dict(xgb_params, silent=1),
                dtrain, num_boost_round=num_boost_rounds)
            print("time to train ....", (time() - start_t))
        else:
            print("    RandomizedSearchCV  ...")
            model = pahse1_gridSearch(X_TR, y_TR)

        dholdout = xgb.DMatrix(X_holdout)
        y_holdout_pred = model.predict(dholdout)
        loss = metrics.mean_absolute_error(y_holdout,y_holdout_pred)
        print("-"*30)
        print("")
        print("   MAE on training Validation (X_holdout) .. %.7f" % loss)

        for i in range(len(test_dates)):
            print("-"*30)
            print(" set test dates .....", test_columns[i])
            YYYY = float(test_columns[i][:4])
            MM = float(test_columns[i][-2:])
            x_test[:,-2] = YYYY
            x_test[:,-1] = MM
            dtest = xgb.DMatrix(x_test)
            print("valuation for x_test .... ")
            y_hat = model.predict(dtest)
            S_test_i[j,:,i] = y_hat


    cv_mean_result = S_test_i.mean(0)

    return cv_mean_result

def makeOutput(data_name,model_i,trainCls, cv_mean_result):

    kaggle_data = trainCls.kaggleSumbmitData()

    print("** cv mean result .... **")
    print(cv_mean_result.shape)
    y_hat_true = kaggle_data["201610"].values
    y_hat = cv_mean_result[:,0]
    loss = metrics.mean_absolute_error(y_hat_true,y_hat)
    print("-"*30)
    print("")
    print("   MAE on Kaggle Submit data.. %.7f" % loss)

    submission = trainCls.getSample()

    test_columns = ['201610','201611','201612','201710','201711','201712']
    for i in range(len(test_columns)):
        pred = cv_mean_result[:,i]

        submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
        print('cv mean result ...', i)

    print( submission.head() )
    ##### WRITE THE RESULTS
    from datetime import datetime

    print( "\nWriting results to disk ..." )
    submission.to_csv('output/xgb_{}_{}_{}.csv'.format(data_name,model_i,datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

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

def pahse1_gridSearch(x_train, y_train):

    y_mean = np.mean(y_train)
    #estimator = lgb.LGBMRegressor(num_leaves=512)
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'nthread' : 4,
        'silent': 1
    }

    clf_xgb = xgb.XGBClassifier(**params)
    print(clf_xgb.get_params().keys() )
    param_grid = {
                    'objective': ['reg:linear'],
                    'eval_metric': ['mae'],
                  'learning_rate': stats.uniform(0.01, 0.03),
                 }
    n_iter_search = 10
    random_search = RandomizedSearchCV(clf_xgb, param_distributions=param_grid,
                                   n_iter=n_iter_search,scoring = two_scorer(),
                                   error_score=0.01 )
    start = time()
    print("-"*30)

    random_search.fit(x_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    #report(random_search.cv_results_)

    print("+")
    print("      Best parameters set found on development set:")
    print(random_search.best_params_)

    return random_search

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")





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
    trainCls = trainClass()

    x_train,y_train,x_test = trainCls.makeTrainDataForXgboost_Normal()
    #y_train = 0.0
    xgb_params_list = readModelClass( np.mean( y_train))

    for i,params_list in enumerate(xgb_params_list):
        res = GridSearch_Valuation(x_train,y_train,x_test,params_list,False)
        #res = np.zeros( (2985217,6) )
        makeOutput("normal",i,trainCls, res)

if __name__ == "__main__":
    main()
