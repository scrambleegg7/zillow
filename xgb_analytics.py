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

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

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
            _df[f][np.isnan(_df[f])] = _df[f].median()
        else:
            print("** object type **")
            _df[f] = _df[f].fillna(-999)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(_df[f].values))
            _df[f] = lbl.transform(list(_df[f].values))

    return _df


def xgbTest(trainCls):

    train2016_df = trainCls.readTrain2016()
    prop2016_df = trainCls.readProp2016()

    df_merged = pd.merge(train2016_df,prop2016_df,on="parcelid",how="left")
    print(df_merged.shape)
    print("-"*30)
    mergedFilterd = fillWithMedian(df_merged)
    #y = mergedFilterd.logerror.values
    _X_df = mergedFilterd.drop(["parcelid", "transactiondate"], axis=1)

    print(_X_df.columns.tolist())

    predictors = [x for x in _X_df.columns.tolist() if x not in [target]]
    param_dist = {
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        #'eval_metric': 'rmse',
        'nthread': 4,
        'silent': True}
    xgb1 = XGBClassifier(**param_dist)

    for i in range(10):
        print("-"*30)
        print(" evaluate counter :" , i)
        rnd = np.random.permutation( range(_X_df.shape[0])  )
        rnd = rnd[:4000]
        _X_df_r = _X_df.ix[rnd]

        msk = np.random.rand(len(_X_df_r)) < 0.5
        train = _X_df_r[msk]
        test = _X_df_r[~msk]
            #learning_rate =0.1,
            #n_estimators=1000,
            #max_depth=5,
            #min_child_weight=1,
            #gamma=0,
            #subsample=0.8,
            #colsample_bytree=0.8,
            #objective= 'reg:linear',
            #objective= 'binary:logistic',
            #nthread=4,
            #scale_pos_weight=1,
            #seed=27)
        modelfit(xgb1, train, test, predictors, useTrainCV=False)

def modelfit(alg, dtrain, dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):


    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds, show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    print("-"*30)
    print(".. start to training ..... ")
    start_t = time()
    alg.fit(dtrain[predictors], dtrain[target],eval_metric=['rmse'])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    print("-"*30)
    print(" Time to train ..  %.4f" %  ( time()-start_t ) )
    #Print model report:
    print "\nModel Report"
    print("-"*30)
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp[:15])


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

def main():
    trainCls = trainClass(test=True)
    xgbTest(trainCls)

if __name__ == "__main__":
    main()
