#
from trainClass import trainClass

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn import model_selection, preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from time import time
import seaborn as sns

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

from statsmodels.graphics.gofplots import qqplot_2samples

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

# target columns name
target = "logerror"

def fillWithMedian(df):
    _df = df.copy()

    missingValueColumns = _df.columns[_df.isnull().any()].tolist()

    for f in missingValueColumns:
        print("...processing fill median...", f)
        if _df[f].dtypes !='object':
            #_df[f][np.isnan(_df[f])] = _df[f].median()
            _df[f][np.isnan(_df[f])] = _df[f].fillna(-1)
        else:
            print("** object type **")
            _df[f] = _df[f].fillna(-999)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(_df[f].values))
            _df[f] = lbl.transform(list(_df[f].values))

    return _df


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

def DataPreparation(trainCls,select_data=False):

    train2016_df = trainCls.readTrain2016()
    prop2016_df = trainCls.readProp2016()

    df_merged = pd.merge(train2016_df,prop2016_df,on="parcelid",how="left")
    print(df_merged.shape)
    print("-"*30)
    mergedFilterd = fillWithMedian(df_merged)

    if select_data:
        mergedFilterd = selectData( mergedFilterd )

    return mergedFilterd

def xgbParamCV(trainCls):

    train2016_df = trainCls.readTrain2016()
    prop2016_df = trainCls.readProp2016()

    df_merged = pd.merge(train2016_df,prop2016_df,on="parcelid",how="left")
    print(df_merged.shape)
    print("-"*30)
    mergedFilterd = fillWithMedian(df_merged)
    _X_df = selectData( mergedFilterd )

    print("-"*30)
    print("select columns")
    print(_X_df.columns.tolist())

    predictors = [x for x in _X_df.columns.tolist() if x not in [target]]

    param_dist = {
        'learning_rate': 0.05,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'min_child_weight': 6,
        'gamma': 0,
        'nthread': 4,
        'silent': True}
    xgb1 = XGBClassifier(**param_dist)

    #Grid seach on subsample and max_features
    #Choose all predictors except target & IDcols
    param_test2 = {
        'max_depth':[3,10,2],
        'min_child_weight':[1,6,2]
    }
    param_test3 = {
        'gamma':[ i / 10. for i in range(0,5) ]
    }

    gsearch2 = GridSearchCV(estimator = xgb1,
                param_grid = param_test3, scoring = two_scorer(),
                n_jobs=4,iid=False, cv=4)

    rnd = np.random.permutation( range(_X_df.shape[0])  )
    rnd = rnd[:4000]
    _X_df_r = _X_df.ix[rnd]

    msk = np.random.rand(len(_X_df_r)) < 0.5
    train = _X_df_r[msk]
    test = _X_df_r[~msk]

    gsearch2.fit(train[predictors],train[target])

    for item in gsearch2.grid_scores_:
        print "%s %s %s" % ('GRIDSCORES\t',"RMSE",item)
    print("Grid best params:",gsearch2.best_params_)
    print("Grid best score:",gsearch2.best_score_)


def model_data_prep(train_df):
    pass

def model_run(train_df):

    x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    y_train = train_df["logerror"].values.astype(np.float32)
    y_mean = np.mean(y_train)
    y_test = y_train

    print('After removing outliers:')
    print("Training Shape", x_train.shape)

    print('Shape train: {}\n'.format(x_train.shape))

    # xgboost params
    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_train)

    #xgb_params = {
    #    'eta': 0.06,
    #    'max_depth': 5,
    #    'subsample': 0.77,
    #    'objective': 'reg:linear',
    #    'eval_metric': 'mae',
    #    'base_score': y_mean,
    #    'silent': 1
    #}
    xgb_params = {
        'eta': 0.06,
        'max_depth': 3,
        'subsample': 0.77,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'min_child_weight': 6,
        'base_score': y_mean,
        'silent': 1,
        'nthread': 4
    }
    cv_result = xgb.cv(xgb_params,
                       dtrain,
                       nfold=5,
                       num_boost_round=200,
                       early_stopping_rounds=50,
                       verbose_eval=10,
                       show_stdv=False
                      )
    num_boost_rounds = len(cv_result)
    print("number of boost rounds..",num_boost_rounds)
    # train model
    print("-"*30)
    print(" training .......")
    start_t = time()
    model = xgb.train(dict(xgb_params, silent=1),
        dtrain, num_boost_round=num_boost_rounds)
    print("time to train ....", (time() - start_t))

    return model

def model_predict(x_test,):

    y_pred = model.predict(dtest)
    print("-"*30)
    print("y_predict (target) shape ", y_pred.shape)
    residual = y_test - y_pred
    train_df["residual"] = residual

    loss = metrics.mean_squared_error(y_test,y_pred)
    print("RMSE : %.7f" % np.sqrt( loss ))
    print("MSE : %.7f" % loss)

    return train_df

def xgbcvtest(train_df):

    print("-"*30)
    print("columns name to be trained ..")
    print(train_df.columns.tolist())

    #x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

    # drop out ouliers
    train_df=train_df[ train_df.logerror > -0.4 ]
    train_df=train_df[ train_df.logerror < 0.4 ]
    train_df.reset_index(drop=True,inplace=True)

    _x_train=train_df.drop(['parcelid','logerror','transactiondate'], axis=1)
    _y_train = train_df["logerror"].values.astype(np.float32)

    train_length = _x_train.shape[0]
    rnd = np.random.permutation( range(train_length)  )
    rnd5 = rnd[:80000]
    rnd5_n = rnd[-10000:]
    x_train = _x_train
    y_train = _y_train
    y_mean = np.mean(y_train)
    print("train size : x y y_mean:",x_train.shape,y_train.shape,y_mean.shape)

    x_test = _x_train
    y_test = _y_train
    print("test size :", x_test.shape)

    xgb_params = {
        'eta': 0.06,
        'max_depth': 3,
        'subsample': 0.77,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'min_child_weight': 6,
        'base_score': y_mean,
        'silent': 1,
        'nthread': 4
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    cv_result = xgb.cv(xgb_params,
                       dtrain,
                       nfold=5,
                       num_boost_round=200,
                       early_stopping_rounds=50,
                       verbose_eval=10,
                       show_stdv=False
                      )
    num_boost_rounds = len(cv_result)
    print(num_boost_rounds)
    # train model
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    y_pred = model.predict(dtest)

    residual = y_test - y_pred
    train_df["residual"] = residual

    return train_df


def modelfit(alg, dtrain, dtest, predictors, target, cv_folds=5, early_stopping_rounds=50):

    #Fit the algorithm on the data
    print("-"*30)
    print(".. start to training w/o adjusting ..... ")
    start_t = time()
    alg.fit(dtrain[predictors], dtrain[target],eval_metric=['rmse'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]

    loss = metrics.mean_squared_error(dtest[target].values,dtest_predictions)
    print("RMSE : %.6f" % np.sqrt( loss ))

    residual = dtest[target].values - dtest_predictions
    return residual

def result():

    #structuretaxvaluedollarcnt      81469
    #lotsizesquarefeet               79623
    #landtaxvaluedollarcnt           77502
    #longitude                       76585
    #latitude                        76387
    #taxamount                       75563
    #calculatedfinishedsquarefeet    68538
    #taxvaluedollarcnt               62746
    #yearbuilt                       62136
    #finishedsquarefeet12            54338
    #regionidzip                     53569
    #regionidcity                    40077
    #rawcensustractandblock          37573
    #propertyzoningdesc              35595
    #censustractandblock             29577

    pass

def data_manage():
    trainCls = trainClass(test=True)
    data_df = DataPreparation(trainCls)

    #
    print("-"*30)
    print(" omit 0.4 to remove outliers data ..")
    data_df=data_df[ abs(data_df.logerror) < 0.4 ]
    data_df.reset_index(drop=True,inplace=True)

    _train_df = model_run(data_df)

    # _train_df incl. residual columns
    residual_df = _train_df[abs(_train_df.residual) > 0.05]
    residual_df.reset_index(drop=True,inplace=True)

    print("-"*30)
    print("residual > 0.05 data shape..", residual_df.shape)

#    _residual_df = residual_df.drop(['residual'], axis=1)
#    _residual_df.reset_index(drop=True,inplace=True)

    _train_df = model_run(residual_df)

#    xgbParamCV(trainCls)

def main():
    data_manage()

if __name__ == "__main__":
    main()
