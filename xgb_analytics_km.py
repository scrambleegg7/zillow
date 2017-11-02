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
from sklearn.cluster import KMeans

from statsmodels.graphics.gofplots import qqplot_2samples

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

# target columns name
target = "logerror"
geoColumns = ["latitude","longitude"]

def fillWithMedian(df):
    _df = df.copy()

    missingValueColumns = _df.columns[_df.isnull().any()].tolist()

    for f in missingValueColumns:
        print("...processing fill median...", f)
        if _df[f].dtypes !='object':
            if f in geoColumns:
                median_ = _df[f].median()
                _df[f] = _df[f].fillna(median_)
            else:
                _df[f] = _df[f].fillna(-1)
        else:
            _df[f] = _df[f].fillna(-1)
            print("** object type **")
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

    _prop2016_df = fillWithMedian(prop2016_df)
    print("original properties shape:", _prop2016_df.shape)

    mergedFilterd = pd.merge(train2016_df,_prop2016_df,on="parcelid",how="left")
    print("train data shape:",mergedFilterd.shape)

    pmerged_df = pd.merge(train2016_df,_prop2016_df,on="parcelid",how="right")
    pmerged_df = pmerged_df[ pmerged_df["logerror"].isnull()  ]
    print("properties null logerror", pmerged_df.shape)

    del train2016_df
    del prop2016_df
    del _prop2016_df

    print("-"*30)
    if select_data:
        mergedFilterd = selectData( mergedFilterd )

    return mergedFilterd, pmerged_df

def kmeanClustering(df,clusters=8):
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(df)
    y_label = kmeans.labels_

    return y_label


def kmClassification(_x_train,_y_train):

    x_train = _x_train.copy()
    y_train = _y_train.copy()

    print("-"*30)
    print(" kmean labeling")
    y_label = kmeanClustering(x_train)

    selectCols = ["latitude","longitude"]
    X_ = x_train[selectCols].values
    y_label = y_label.ravel()

    print("train,test shape",X_.shape,y_label.shape)

    dtrain = xgb.DMatrix(X_, label=y_label)
    #xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 8

    cv_result = xgb.cv(param,
                   dtrain,
                   nfold=5,
                   num_boost_round=200,
                   early_stopping_rounds=50,
                   verbose_eval=10,
                   show_stdv=False
                  )
    num_boost_rounds = len(cv_result)
    print("-"*30)
    print("start to train")
    start_t = time()
    model = xgb.train(param, dtrain, num_boost_round=num_boost_rounds)
    print("end train", ( time() - start_t ) )

    print("-"*30)
    #dtest = xgb.DMatrix(X_)
    #pred = bst.predict(dtest)
    #error_rate = np.sum(pred != y_label) / np.float( len(y_label) )
    #print('Test error using softmax = %.6f' %error_rate)

    #temp_df = x_train[selectCols].copy()
    #temp_df["klabel"] = pred

    #g = sns.FacetGrid(temp_df, col="klabel")
    #g.map(plt.scatter, "latitude","longitude").add_legend()
    #    plt.show()

    return model

def model_run(_x_train,_y_train):

    #x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    x_train = _x_train.copy()
    y_train = _y_train.copy()

    print("-"*30)
    print(" kmean labeling")
    y_label = kmeanClustering(x_train)

    #x_train["klabel"] = y_label

    mask = (y_label == 1) | (y_label == 3)
    #x_train = x_train[mask].drop(['klabel'],axis=1)
    #y_ = train_df["logerror"]
    #y_train = y_train[mask].values.astype(np.float32)

    y_mean = np.mean(y_train)

    print('After removing outliers:')
    print("Training Shape", x_train.shape)
    print('label shape: {}\n'.format(y_train.shape))

    # xgboost params
    dtrain = xgb.DMatrix(x_train, y_train)

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

def getXtrain(train_df):
    x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    return x_train

def getY_target(_df,target="logerror"):
    y_ = _df[target]
    #y_train = y_[mask].values.astype(np.float32)
    return y_

def model_predict(model,dtest):

    y_hat = model.predict(dtest)
    print("-"*30)
    print(" model prediction ....")
    return y_hat

def showMSE(y_test,y_pred):
    print("y_predict (target) shape ", y_pred.shape)
    residual = y_test - y_pred
    #train_df["residual"] = residual

    loss = metrics.mean_squared_error(y_test,y_pred)
    print("RMSE : %.7f" % np.sqrt( loss ))
    print("MSE : %.7f" % loss)

def showResidualDistribution(y_test,y_pred):
    residual = y_test - y_pred
    sns.distplot(residual)
    plt.show()

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

def propertiesTest(model,_prop2016_df):

    pmerged_df = pd.merge(train,properties,on="parcelid",how="right")

def data_manage_2():
    trainCls = trainClass(test=True)

    data_df, prop2016_df = DataPreparation(trainCls)

    #
    print("-"*30)
    print(" omit 0.4 to remove outliers data ..")
    data_df=data_df[ abs(data_df.logerror) < 0.4 ]
    data_df.reset_index(drop=True,inplace=True)

    _x_train = data_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    _y_train = getY_target(data_df)

    model = kmClassification(_x_train,_y_train)

    #
    # properties data frame
    #
    #x_test = prop2016_df.drop(['parcelid', 'logerror','transactiondate'], axis=1).copy()
    #print("-"*30)
    #print("properties test shape",x_test.shape)

    x_test = prop2016_df[geoColumns].copy()
    print(" test data shape",x_test.shape)

    dtest = xgb.DMatrix(x_test.values)
    print("-"*30)
    print(" predict kmean label based ")
    pred = model.predict(dtest)

    x_test["klabel"] = pred

    g = sns.FacetGrid(x_test, col="klabel")
    g.map(plt.scatter, "latitude","longitude").add_legend()
    plt.show()



def data_manage():
    trainCls = trainClass(test=True)

    data_df, prop2016_df = DataPreparation(trainCls)

    #
    print("-"*30)
    print(" omit 0.4 to remove outliers data ..")
    data_df=data_df[ abs(data_df.logerror) < 0.4 ]
    data_df.reset_index(drop=True,inplace=True)

    #
    #
    # training data drop id logerror, transactiondate
    #
    #
    _x_train = data_df.drop(['parcelid','logerror','transactiondate'], axis=1)
    _y_train = getY_target(data_df)


    model = model_run(_x_train,_y_train)
    print("after running model shape:", _x_train.shape)
    # set data_df (fulll dataset) for validating
    #y_val = getY_target(data_df)
    # preparation for valid data
    dtest = xgb.DMatrix(_x_train)
    y_hat = model_predict(model, dtest)

    showMSE(_y_train,y_hat)
    showResidualDistribution(_y_train,y_hat)



def main():
    data_manage_2()

if __name__ == "__main__":
    main()
