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





def get_features1(df):
    #df.loc[:,"transactiondate"] = pd.to_datetime(df["transactiondate"])
    df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )
    df = df.assign( transactiondate_year = lambda x: x.transactiondate.dt.year   )
    df = df.assign( Month = lambda x: x.transactiondate.dt.month   )
    df = df.assign( transactiondate = lambda x: x.transactiondate.dt.quarter   )
    df = df.fillna(-1.0)
    return df



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

    estimator = lgb.LGBMRegressor(num_leaves=512)

    # run randomized search

    param_grid = {
        'learning_rate': stats.uniform() ,
        'num_leaves': [512],
        'max_depth': stats.randint(low=3,high=10),
        'max_bin' : stats.randint(low=10,high=70),
        'boosting_type' : ['gbdt'],
        'objective' : ['regression'],
        'metric' : ['l1'],
        'bagging_fraction' : stats.uniform(),
        'bagging_freq' : stats.randint(low=10,high=100),
        'sub_feature' : stats.uniform(),
        'min_data' : stats.randint(low=100,high=1000),
        'min_hessian' : stats.uniform(),
        'feature_fraction_seed' : [2],
        'bagging_seed' : [3]

    }
    n_iter_search = 20
    random_search = RandomizedSearchCV(estimator, param_distributions=param_grid,
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



def gridSearchCVL_LGB(GridSearchSW=False,add_data_sw=0):

    trainCls = trainClass(test=True)

    params_list = modelCls.getModelParams()
    params_list = [ v for (k,v) in params_list.items() if "lightgbm" in k ]

    kaggle_data = trainCls.kaggleSumbmitData()
    x_train, y_train, x_test = trainCls.makeTrainDataForLightGBM(add_data_sw)

    params_list = modelCls.getModelParams()
    params_list = [ v for (k,v) in params_list.items() if "lightgbm" in k ]
    row_num, feat_num = x_train.shape
    _n_splits = 5
    kf = KFold(n_splits=_n_splits, shuffle=True,random_state=42)
    print("   number of folds .... 5")
    print("Sample test shape", x_test.shape)
    base_models_length = len(params_list)

    params = params_list[0]

    test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
    test_columns = ['201610','201611','201612','201710','201711','201712']
    S_test_i = np.zeros( (_n_splits,  x_test.shape[0], len(test_columns)) )

    for j, (train_idx, test_idx) in enumerate( kf.split(x_train) ):
        X_train = x_train[train_idx]
        y_true = y_train[train_idx]
        X_holdout = x_train[test_idx]
        y_holdout = y_train[test_idx]

        print("-"*30)
        print("       FOLD : %d " % j)
        print("       train shape ......")
        print(X_train.shape,y_true.shape)

        if not GridSearchSW:
            d_train = lgb.Dataset(X_train, label=y_true)
            print("\nFitting LightGBM model ...")
            clf = lgb.train(params, d_train, 430)
        else:
            print("    RandomizedSearchCV  ...")
            clf = pahse1_gridSearch(X_train, y_true)

        y_holdout_pred = clf.predict(X_holdout)
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
            print("valuation for x_test .... ")
            y_hat = clf.predict(x_test)
            S_test_i[j,:,i] = y_hat

    cv_mean_result = S_test_i.mean(0)
    print("** cv mean result .... **")
    print(cv_mean_result.shape)
    y_hat_true = kaggle_data["201610"].values
    y_hat0 = cv_mean_result[:,0]
    loss = metrics.mean_absolute_error(y_hat_true,y_hat)
    print("-"*30)
    print("")
    print("   MAE on Kaggle Submit data.. %.7f" % loss)


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

    gridSearchCVL_LGB(False,2)   # additional features on Build Year
    gridSearchCVL_LGB(False,1)   # additional features
    gridSearchCVL_LGB()

if __name__ == "__main__":
    main()
