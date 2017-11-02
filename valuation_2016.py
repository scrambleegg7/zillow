import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings

import xgboost
import os

from sklearn.model_selection import RandomizedSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold


from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


matplotlib.style.use('ggplot')
warnings.filterwarnings("ignore")

#from ggplot import *

#%matplotlib inline
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from time import time
import xgboost as xgb

from xgboost import plot_tree
from sklearn import cross_validation, metrics

from statsmodels.graphics.gofplots import qqplot_2samples

from trainClass import trainClass

class DataObjects(object):
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        #self.X_val = X_val
    def setMask(self,mask):
        self.X_train_mask = self.X_train[mask]
        self.y_train_mask = self.y_train[mask.values]
    def getXtrainMask(self):
        return self.X_train_mask
    def getytrainMask(self):
        return self.y_train_mask
    def setValMask(self,mask):
        self.X_val_mask = self.X_train[mask]
        self.y_val_mask = self.y_train[mask.values]

class propertiesCounterClass(object):

    def __init__(self):

        self.zip_count = None
        self.city_count = None
        self.medyear = None
        self.meanarea = None
        self.medlat = None
        self.medlong = None
        self.zipstd = None
        self.citystd = None
        self.hoodstd = None
        self.d_median = None
        self.region1 = None
        self.build1 = None
        self.fips1 = None

    def setUpCounter(self,_df):
        # Number of properties in the zip
        self.zip_count = _df['region_zip'].value_counts().to_dict()
        # Number of properties in the city
        self.city_count = _df['region_city'].value_counts().to_dict()
        # Median year of construction by neighborhood
        self.medyear = _df.groupby('region_neighbor')['build_year'].aggregate('median').to_dict()
        # Mean square feet by neighborhood
        self.meanarea = _df.groupby('region_neighbor')['area_total_calc'].aggregate('mean').to_dict()
        # Neighborhood latitude and longitude
        self.medlat = _df.groupby('region_neighbor')['latitude'].aggregate('median').to_dict()
        self.medlong = _df.groupby('region_neighbor')['longitude'].aggregate('median').to_dict()


def setFeaturesSumAndVar(_df):

    properties=_df.copy()
    print("\n")
    print("-"*30)
    print("\n    set Sum/Var of Features of properties ...")
    print("-"*30)
    properties["features_sum"] = properties.sum(axis=1).values.reshape(-1,1)
    properties["features_var"] = properties.var(axis=1).values.reshape(-1,1)

    scaler = StandardScaler()
    target_list = ["features_sum","features_var"]
    for c in target_list:
        properties[c] = scaler.fit_transform(properties[c].values.reshape(-1,1))

    del _df
    return properties

def changeColNames(_df):

    #train_df = self.getTrain()
    #train_df = train_df.rename( columns = {'parcelid':'id_parcel'})
    #self.setTrain(train_df)

    prop2016_df = _df.copy()
    prop_df = prop2016_df.rename(
        columns = {
            "yearbuilt":"build_year",
            "basementsqft":"area_basement",
            "yardbuildingsqft17":"area_patio",
            "yardbuildingsqft26":"area_shed",
            "poolsizesum":"area_pool",
            "lotsizesquarefeet":"area_lot",
            "garagetotalsqft":"area_garage",
            "finishedfloor1squarefeet":"area_firstfloor_finished",
            "calculatedfinishedsquarefeet":"area_total_calc",
            "finishedsquarefeet6":"area_base",
            "finishedsquarefeet12":"area_live_finished",
            "finishedsquarefeet13":"area_liveperi_finished",
            "finishedsquarefeet15":"area_total_finished",
            "finishedsquarefeet50":"area_unknown",
            "unitcnt":"num_unit",
            "numberofstories":"num_story",
            "roomcnt":"num_room",
            "bathroomcnt":"num_bathroom",
            "bedroomcnt":"num_bedroom",
            "calculatedbathnbr":"num_bathroom_calc",
            "fullbathcnt":"num_bath",
            "threequarterbathnbr":"num_75_bath",
            "fireplacecnt":"num_fireplace",
            "poolcnt":"num_pool",
            "garagecarcnt":"num_garage",
            "regionidcounty":"region_county",
            "regionidcity":"region_city",
            "regionidzip":"region_zip",
            "regionidneighborhood":"region_neighbor",
            "taxvaluedollarcnt":"tax_total",
            "structuretaxvaluedollarcnt":"tax_building",
            "landtaxvaluedollarcnt":"tax_land",
            "taxamount":"tax_property",
            "assessmentyear":"tax_year",
            "taxdelinquencyflag":"tax_delinquency",
            "taxdelinquencyyear":"tax_delinquency_year",
            "propertyzoningdesc":"zoning_property",
            "propertylandusetypeid":"zoning_landuse",
            "propertycountylandusecode":"zoning_landuse_county",
            "fireplaceflag":"flag_fireplace",
            "hashottuborspa":"flag_tub",
            "buildingqualitytypeid":"quality",
            "buildingclasstypeid":"framing",
            "typeconstructiontypeid":"material",
            "decktypeid":"deck",
            "storytypeid":"story",
            "heatingorsystemtypeid":"heating",
            "airconditioningtypeid":"aircon",
            "architecturalstyletypeid":"architectural_style"
         }
    )

    print("")
    print("-"*40)
    print("     Check Columns names after changing to realistic ones .......")
    print(prop_df.columns.values)

    del _df
    return prop_df

def changeObjects(_df):
    properties = _df.copy()

    #droplist = ["propertyzoningdesc","propertycountylandusecode","fireplaceflag"]
    #properties.drop(droplist, axis=1, inplace=True)
    print(" LabelEncoder ..")
    # use LabelEncoder for object
    for c in properties.columns:
        #properties[c]=properties[c].fillna(-1)
        if properties[c].dtype == 'object':
            print("\nColumns ..... %s" % c)
            print(" change to float and fill with 0..")
            properties[c].fillna(0, inplace=True)

            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            properties[c] = properties[c].astype(np.float32)

    #self.setProp(properties)
    return properties

def changeGeoData(_df):
    properties = _df.copy()

    print("\n")
    print("-"*40)
    print("\n      geo data divide 1e6")

    properties["latitude"] = properties["latitude"].apply(lambda x:x / 1e6)
    properties["longitude"] = properties["longitude"].apply(lambda x:x / 1e6)

    properties["over_under_range_latitude"] = properties["latitude"].apply(lambda x:1 if x >= 33.8 and x <= 34.15 else 0 )
    properties["over_under_range_longitude"] = properties["longitude"].apply(lambda x:1 if x >= -118.5 and x <= -118.25 else 0 )

    #self.setProp(properties)
    del _df
    return properties

def changeDataTypes(_df):
    #
    # change data type from float64 to float32
    #
    properties = _df.copy()
    print("\n")
    print("-"*40)
    print( "\n** change data types float64 --> float32 ...")
    for col in properties.columns:
        if properties[col].dtype != object:
            if properties[col].dtype == float:
                properties[col] = properties[col].astype(np.float32)

    del _df
    return properties

def split_buildyear(_df):

    properties = _df.copy()
    #properties["build_year"].fillna(properties["build_year"].mean() ,inplace=True)

    bins = np.linspace(properties.build_year.min(),properties.build_year.max(),6)
    #pd.DataFrame( np.digitize(properties.build_year,bins[1:-1]), columns=["build_cat"]).build_cat.value_counts()

    bins = [1900,1930,1970,2000]
    print("* split build year...",bins)
    print("* omit first / last item to ")

    properties["build_category"] = np.digitize(properties.build_year,bins)
    #properties.build_category.value_counts()

    del _df
    return properties

def changeTrainYearMonthFromTransactionDate(_df):

    df = _df.copy()

    print("\n")
    print("-"*40)
    print("\n      Changing train transactiondate --> Year / Month / Day / Quarter ...")
    df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )
    df = df.assign( transaction_year = lambda x: x.transactiondate.dt.year   )
    df = df.assign( transaction_month = lambda x: x.transactiondate.dt.month   )
    df = df.assign( transaction_day = lambda x: x.transactiondate.dt.day   )

    df = df.assign( transaction_qtr = lambda x: x.transactiondate.dt.quarter   )
    #df = df.assign( transaction_qtrstr = lambda x: x.transactiondate.dt.to_period("Q")   )
    #df = df.drop(['transactiondate'], axis=1)
    #self.setTrain(df)
    del _df
    return df

def calculate_features(_df, propCnt):
    df = _df.copy()
    # Nikunj's features
    # Number of properties in the zip
    df['N-zip_count'] = df['region_zip'].map(propCnt.zip_count)
    # Number of properties in the city
    df['N-city_count'] = df['region_city'].map(propCnt.city_count)
    # Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['num_garage']>0) & \
                         (df['pooltypeid10']>0) & \
                         (df['aircon']!=5))*1

    # More features
    # Mean square feet of neighborhood properties
    df['mean_area'] = df['region_neighbor'].map(propCnt.meanarea)
    # Median year of construction of neighborhood properties
    df['med_year'] = df['region_neighbor'].map(propCnt.medyear)
    # Neighborhood latitude and longitude
    df['med_lat'] = df['region_neighbor'].map(propCnt.medlat)
    df['med_long'] = df['region_neighbor'].map(propCnt.medlong)

    df['zip_std'] = df['region_zip'].map(propCnt.zipstd)
    df['city_std'] = df['region_city'].map(propCnt.citystd)
    df['hood_std'] = df['region_neighbor'].map(propCnt.hoodstd)

    del _df
    return df

def training():
    GPU_SW = 1
#

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))

    xgbparams = {
                'eta': 0.037,
                'max_depth': 10,
                'subsample': 0.7,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'lambda': 0.9,
                'alpha': 0.9,
                'silent': 1
            }

    if GPU_SW:
        xgbparams['gpu_id'] = 0
        xgbparams['max_bin'] = 16
        xgbparams['tree_method'] = 'gpu_exact'
    xgparam = list(xgbparams.items())

    my_fips_codes = train_df.fips.unique().astype(int)
    X_select = X[num_col_list]

    # now, select SEPT transaction month for validation
    mask_val = X_select.transaction_month == 9
    X_val_copy = X_select[mask_val]
    y_val_copy = y[mask_val.values]
    y_hat = np.zeros_like(y_val_copy )
    print("just only Sept validatation data shape...", y_hat.shape, X_val_copy.shape)

    y_sept = y[mask_val.values]


    for fips_code in my_fips_codes:
        fips_mask = X_select.fips == fips_code
        X_copy = X_select[fips_mask ]
        y_copy = y[fips_mask.values]

        fips_mask_val = X_val_copy.fips == fips_code
        x_val = X_val_copy[fips_mask_val]
        y_val = y_val_copy[fips_mask_val.values]

        print("      FIPS code ....", fips_code)
        dataObjects = DataObjects(X_copy,y_copy)
        print(X_copy.shape,y_copy.shape)
        print("feature numbers", len(X_copy.columns.tolist()) )

        #
        # set september data shape for valid.....
        #

        rows,cols = x_val.shape

        #last_y_pred = np.zeros( len(y)  )
        month_preds = np.zeros( (rows,8)   )

        for m in range(8):
            target_month = m+1
            val_month = 9

            # split train and valuation..
            mask = X_copy.transaction_month <= target_month
            #mask_val = X_copy.transaction_month == val_month

            dataObjects.setMask(mask)
            x_train, y_train = dataObjects.X_train_mask, dataObjects.y_train_mask
            #print("   trainig shape : "    ,x_train.shape,y_train.shape)
            rmse_ = rmse_cv(rfc,x_train,y_train)
            #print(m+1,"RF : " , np.mean( rmse_ ))

            rmse_ = rmse_cv(gfc,x_train,y_train)
            #print(m+1,"GB : ", np.mean( rmse_ ))

            rfc.fit(x_train,y_train)
            gfc.fit(x_train,y_train)

            x_train1 = np.zeros( (x_train.shape[0],3 ) )
            x_train1[:,0] = rfc.predict(x_train).ravel()
            x_train1[:,1] = gfc.predict(x_train).ravel()

            #model_elastic.fit(x_train,y_train)

            # xgb
            #x_tr, x_v, y_tr, y_v = train_test_split(x_train,y_train)
            dtrain = xgb.DMatrix(x_train, y_train)
            #dval = xgb.DMatrix(x_v,y_v)

            #dataObjects.setValMask(mask_val)
            #x_val,y_val = dataObjects.X_val_mask, dataObjects.y_val_mask
            print("   validation shape : "    ,x_val.shape,y_val.shape)
            dval = xgb.DMatrix(x_val, y_val)

            #dtest = xgb.DMatrix(x_v)
            watchlist  = [ (dtrain,'train'),(dval,'val')]
            xgparam.extend( {'base_score':np.mean(y_train) }.items()  )
            bst = xgb.train(params=xgparam,dtrain=dtrain,
                            num_boost_round=1500,evals=watchlist,early_stopping_rounds=300,
                            verbose_eval=False)
            #bst = xgb.train(params=xgparam,dtrain=dtrain,
            #                num_boost_round=1200,
            #                verbose_eval=False)

            x_train1[:,2] = bst.predict(xgb.DMatrix(x_train), ntree_limit = bst.best_ntree_limit)
            model_lasso.fit(x_train1,y_train)
            #
            #


            preds = []
            losses = []
            names = []
            for k, (model,name) in  enumerate([(rfc,'Rforest'),(gfc,'Gboost'),
                                               (bst,'xgb')       ]):
                names.append(name)
                if name != "xgb":
                    pred = model.predict(x_val)
                    preds.append(pred)
                else:
                    # xgb
                    dval = xgb.DMatrix(x_val)
                    pred = model.predict(dval,ntree_limit = bst.best_ntree_limit)
                    preds.append(pred)

                loss = metrics.mean_absolute_error(y_val,pred)
                losses.append(loss)

            min_index = np.argmin( np.array(losses) )
            print(" train month:%d   minimum loss:%.6f model:%s" % (target_month,
                                                                    losses[min_index], names[min_index]))


            val_mtx = np.zeros( (x_val.shape[0],3) )
            for i, pred in enumerate(preds):
                val_mtx[:,i] = pred.ravel()

            y_predict_elastic = model_lasso.predict(val_mtx)
            loss = metrics.mean_absolute_error(y_val,y_predict_elastic)
            print("    Elastic stacking loss ", loss)

            #    month_preds[:,m] = preds[min_index].ravel()
            month_preds[:,m] = y_predict_elastic


            #y_pred_val_rf = rfc.predict(x_val)
            #y_pred_val_gf = gfc.predict(x_val)
            #y_pred_xgb = bst.predict(dval)

            #loss_r = metrics.mean_absolute_error(y_val,y_pred_val_rf)
            #loss_g = metrics.mean_absolute_error(y_val,y_pred_val_gf)
            #loss_bst = metrics.mean_absolute_error(y_val,y_pred_xgb)

            #sign_mask_rf = np.sign(y_val) != np.sign(y_pred_val_rf)
            #sign_mask_gf = np.sign(y_val) != np.sign(y_pred_val_gf)

            #print("wrong sign RF ",  np.sum(sign_mask_rf) / np.float( y_val.shape[0] )     )
            #print("wrong sign Gradient ",  np.sum(sign_mask_gf) / np.float( y_val.shape[0] )     )
            #x_val["y_pred"] = y_pred_val
            #model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_val, y_val)
            #y_pred_val_lasso = model_lasso.predict(x_val)

            #loss_lasso = metrics.mean_absolute_error(y_val,y_pred_val_lasso)
            #print(m+1, loss_train, val_month , loss_lasso)

            #sign_mask = np.sign(y_val) != np.sign(y_pred_val_lasso)
            #print("   wrong sign ratio --> " ,   np.sum(sign_mask) / np.float(len(y_pred_val_lasso))  )
            #x_val = x_val[sign_mask]
            #y_val = y_val[sign_mask]
            #y_pred_val = rfc.predict(x_val)
            #loss = metrics.mean_absolute_error(y_val,y_pred_val * -1)
            #print("   revalue with wrong sign data ....", loss)

        y_mean = month_preds.mean(1)
        #print("fips code and y_pred shape : ", fips_code,y_pred_val_gf.shape)
        #fips_y_pred.extend(y_pred_val_gf.ravel())
        #fips_y_pred.extend(y_mean.ravel())
        #y_sept.extend(y_val.ravel())
        y_hat[ fips_mask_val.values ] = y_mean.ravel()


    plt.plot(range(len(y_sept)),y_sept ,c="r")
    plt.plot(range(len(y_sept)),y_hat,c="b")
    plt.show()

    print(metrics.mean_absolute_error(y_sept,y_hat)   )

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))

def valuation_2017(train_df, prop_df, propCounterCls):

    print(" *** Making X y for training validation ***")
    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    X = train_df.drop(["parcelid","transactiondate","logerror"], axis=1)
    X.reset_index(drop=True,inplace=True)
    #y_abs  = train_df.abs_logerror.values.astype(np.float32)
    y = train_df.logerror.values.astype(np.float32)

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    num_col_list = [c for c in X.columns.tolist() if "num" in c or "transaction" in c or "mean" in c or "med" in c]
    num_col_list.extend(["fips","N-zip_count","N-city_count","N-GarPoolAC","zip_std","city_std","hood_std"])
    print(num_col_list)

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    print(" ** loading model parameters ...")
    print(" GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier")

    gfc = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03,subsample=0.7)
    rfc = RandomForestRegressor(n_estimators=100,n_jobs=-1)
    gclf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,subsample=0.6)

    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
    model_ridge = RidgeCV(alphas = [1, 0.1, 0.001, 0.0005])
    model_elastic = ElasticNetCV(alphas = [1, 0.1, 0.001, 0.0005])

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))

    #test_dates = ['2017-10-01','2017-11-01','2017-12-01']
    #test_columns = ['201710','201711','201712']

    test_dates = ['2016-10-01','2016-11-01','2016-12-01']
    test_columns = ['201610','201611','201612']

    trainCls = trainClass()
    submission_df = trainCls.readSampleSub()
    print(" submission_df shape......", submission_df.shape)

    X_test = prop_df.copy()
    X_test["transactiondate"] = test_dates[0]
    X_test = changeTrainYearMonthFromTransactionDate(X_test)

    X_test = X_test.fillna(propCounterCls.d_median)

    X_test = pd.merge(X_test,propCounterCls.build1, on=['build_category'], how='left')
    X_test = pd.merge(X_test,propCounterCls.region1, on=['region_county'], how='left')
    X_test = pd.merge(X_test,propCounterCls.fips1, on=['fips'], how='left')

    X_test = calculate_features(X_test, propCounterCls)
    X_test.zip_std = X_test.zip_std.fillna(  X_test.zip_std.median()  )
    X_test.city_std = X_test.city_std.fillna(  X_test.city_std.median()  )
    X_test.hood_std = X_test.hood_std.fillna(  X_test.hood_std.median()  )

    # 2016
    X_test.mean_area = X_test.mean_area.fillna(  X_test.mean_area.median()  )
    X_test.med_year = X_test.med_year.fillna(  X_test.med_year.median()  )


    X_test.reset_index(drop=True,inplace=True)

    GPU_SW = 0
    #

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))

    xgbparams = {
                'eta': 0.037,
                'max_depth': 10,
                'subsample': 0.7,
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'lambda': 0.9,
                'alpha': 0.9,
                'silent': 1
            }

    if GPU_SW:
        xgbparams['gpu_id'] = 0
        xgbparams['max_bin'] = 16
        xgbparams['tree_method'] = 'gpu_exact'
    xgparam = list(xgbparams.items())

    my_fips_codes = train_df.fips.unique().astype(int)
    X_select = X[num_col_list]

    # now, select SEPT transaction month for validation
    mask_val = X_select.transaction_month == 9
    X_val_copy = X_select[mask_val]
    y_val_copy = y[mask_val.values]
    y_hat = np.zeros_like(y_val_copy )
    print("just only Sept validatation data shape...", y_hat.shape, X_val_copy.shape)

    y_sept = y[mask_val.values]

    # X_test
    X_test_copy = X_test[num_col_list]



    for dk, tdate in enumerate(test_dates):

        print("... X_test prediction. target date. --> ",tdate)

        X_test["transactiondate"] = tdate

        X_test = changeTrainYearMonthFromTransactionDate(X_test)
        X_test_select = X_test[num_col_list]
        #y_hat = np.zeros( (X_test_select.shape[0],3)   )
        #X_test_copy = X_test_select[fips_mask ]
        #y_hat_copy = y_hat[fips_mask.values]
        print("X_test & y_hat Shape ..", X_test_select.shape )
        #continue



        for fips_code in my_fips_codes:
            fips_mask = X_select.fips == fips_code
            X_copy = X_select[fips_mask ]
            y_copy = y[fips_mask.values]

            fips_mask_val = X_val_copy.fips == fips_code
            x_val = X_val_copy[fips_mask_val]
            y_val = y_val_copy[fips_mask_val.values]

            print("      FIPS code ....", fips_code)
            dataObjects = DataObjects(X_copy,y_copy)
            print(X_copy.shape,y_copy.shape)
            print("* valuation feature numbers", len(X_copy.columns.tolist()) )

            #
            # set september data shape for valid.....
            #
            fips_mask_test = X_test_select.fips == fips_code
            X_test_copy = X_test_select[fips_mask_test]
            print("* X test size (by fips code)..", X_test_copy.shape )

            rows,cols = X_test_copy.shape
            #last_y_pred = np.zeros( len(y)  )
            test_month_preds = np.zeros( (rows,9)   )

            for m in range(9):
                target_month = m+1

                print("target month....", target_month)

                # split train and valuation..
                mask = X_copy.transaction_month <= target_month
                #mask_val = X_copy.transaction_month == val_month

                dataObjects.setMask(mask)
                x_train, y_train = dataObjects.X_train_mask, dataObjects.y_train_mask
                #print("   trainig shape : "    ,x_train.shape,y_train.shape)
                #rmse_ = rmse_cv(rfc,x_train,y_train)
                #rmse_ = rmse_cv(gfc,x_train,y_train)

                rfc.fit(x_train,y_train)
                gfc.fit(x_train,y_train)

                x_train1 = np.zeros( (x_train.shape[0],3 ) )
                x_train1[:,0] = rfc.predict(x_train).ravel()
                x_train1[:,1] = gfc.predict(x_train).ravel()


                dtrain = xgb.DMatrix(x_train, y_train)
                dval = xgb.DMatrix(x_val, y_val)

                #dtest = xgb.DMatrix(x_v)
                watchlist  = [ (dtrain,'train'),(dval,'val')]
                xgparam.extend( {'base_score':np.mean(y_train) }.items()  )
                bst = xgb.train(params=xgparam,dtrain=dtrain,
                                num_boost_round=1500,evals=watchlist,early_stopping_rounds=300,
                                verbose_eval=False)
                #bst = xgb.train(params=xgparam,dtrain=dtrain,
                #                num_boost_round=1200,
                #                verbose_eval=False)

                x_train1[:,2] = bst.predict(xgb.DMatrix(x_train), ntree_limit = bst.best_ntree_limit)
                model_lasso.fit(x_train1,y_train)
                #
                #


                preds = []
                losses = []
                names = []
                test_mtx = np.zeros( (X_test_copy.shape[0],3) )
                for k, (model,name) in  enumerate([(rfc,'Rforest'),(gfc,'Gboost'),
                                                   (bst,'xgb')       ]):
                    names.append(name)
                    if name != "xgb":
                        #pred = model.predict(x_val)
                        #preds.append(pred)
                        test_mtx[:,k] = model.predict(X_test_copy)

                    else:
                        # xgb
                        dtest = xgb.DMatrix(X_test_copy)
                        test_mtx[:,k] = model.predict(dtest,ntree_limit = bst.best_ntree_limit)
                        #pred = model.predict(dval,ntree_limit = bst.best_ntree_limit)
                        #preds.append(pred)

                print("  test_matrix result size...", test_mtx.shape)
                print("  print first 5 rows of test_matrix Rf, Gb, xgb ...")
                print(test_mtx[:5,:])

                y_predict_elastic = model_lasso.predict(test_mtx)
                test_month_preds[:,m] = y_predict_elastic
                print("  lasso model result size...", y_predict_elastic.shape)
                print("  then show first 10 results lasso model", y_predict_elastic[:10])


                #y_pred_val_rf = rfc.predict(x_val)
                #y_pred_val_gf = gfc.predict(x_val)
                #y_pred_xgb = bst.predict(dval)

                #loss_r = metrics.mean_absolute_error(y_val,y_pred_val_rf)
                #loss_g = metrics.mean_absolute_error(y_val,y_pred_val_gf)
                #loss_bst = metrics.mean_absolute_error(y_val,y_pred_xgb)

                #sign_mask_rf = np.sign(y_val) != np.sign(y_pred_val_rf)
                #sign_mask_gf = np.sign(y_val) != np.sign(y_pred_val_gf)

                #print("wrong sign RF ",  np.sum(sign_mask_rf) / np.float( y_val.shape[0] )     )
                #print("wrong sign Gradient ",  np.sum(sign_mask_gf) / np.float( y_val.shape[0] )     )
                #x_val["y_pred"] = y_pred_val
                #model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_val, y_val)
                #y_pred_val_lasso = model_lasso.predict(x_val)

                #loss_lasso = metrics.mean_absolute_error(y_val,y_pred_val_lasso)
                #print(m+1, loss_train, val_month , loss_lasso)

                #sign_mask = np.sign(y_val) != np.sign(y_pred_val_lasso)
                #print("   wrong sign ratio --> " ,   np.sum(sign_mask) / np.float(len(y_pred_val_lasso))  )
                #x_val = x_val[sign_mask]
                #y_val = y_val[sign_mask]
                #y_pred_val = rfc.predict(x_val)
                #loss = metrics.mean_absolute_error(y_val,y_pred_val * -1)
                #print("   revalue with wrong sign data ....", loss)

            y_mean = test_month_preds.mean(1)
            print("   mean size of test_mtx by rows..", y_mean.shape)
            print("   show first 10 y_mean results...", y_mean[:10])
            #print("fips code and y_pred shape : ", fips_code,y_pred_val_gf.shape)
            #fips_y_pred.extend(y_pred_val_gf.ravel())
            #fips_y_pred.extend(y_mean.ravel())
            #y_sept.extend(y_val.ravel())
            submission_df.loc[ fips_mask_test.values, test_columns[dk] ] = y_mean.ravel()
            print(" submission file size to copy .. ",submission_df.loc[ fips_mask_test.values, test_columns[dk] ].shape)


    print( "\nWriting results to disk ..." )
    submission_df.to_csv('output/sub_2016_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
    print( "\nFinished ...")




def process1():

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))

    trainCls = trainClass()
    #train_2017 = trainCls.readTrain2017()
    #prop_2017 = trainCls.readProp2017()

    train_2017 = trainCls.readTrain2016()
    prop_2017 = trainCls.readProp2016()


    print(datetime.today().strftime("%A %d. %B %Y"))
    prop = setFeaturesSumAndVar(prop_2017)
    print(datetime.today().strftime("%A %d. %B %Y"))
    prop_df = changeColNames(prop)

    print(datetime.today().strftime("%A %d. %B %Y"))
    prop_df = changeObjects(prop_df)

    print(datetime.today().strftime("%A %d. %B %Y"))
    prop_df = changeGeoData(prop_df)

    propCounterCls = propertiesCounterClass()
    propCounterCls.setUpCounter(prop_df)

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    numberOfNullCols = prop_df.isnull().sum(axis=1)
    propCounterCls.d_median = prop_df.median(axis=0)

    prop_df["znull"] = numberOfNullCols

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    prop_df = changeDataTypes(prop_df)

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    prop_df = split_buildyear(prop_df)

    #
    # making training data
    #
    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    print(" *** Make train_df ***")
    train_df = train_2017.merge(prop_df,how="left",on="parcelid")

    train_df = changeTrainYearMonthFromTransactionDate(train_df)
    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    train_df = train_df.fillna(propCounterCls.d_median)

    propCounterCls.citystd = train_df.groupby('region_city')['logerror'].aggregate("std").to_dict()
    propCounterCls.zipstd = train_df.groupby('region_zip')['logerror'].aggregate("std").to_dict()
    propCounterCls.hoodstd = train_df.groupby('region_neighbor')['logerror'].aggregate("std").to_dict()

    b_cat = train_df.groupby(["build_category"],as_index=False)["logerror"].aggregate("mean")
    build1=pd.DataFrame(b_cat)
    build1.columns.values[1] = 'mean_build'
    train_df = pd.merge(train_df,build1, on=['build_category'], how='left')
    propCounterCls.build1 = build1

    r_cat = train_df.groupby(["region_county"],as_index=False)["logerror"].aggregate("mean")
    region1=pd.DataFrame(r_cat)
    region1.columns.values[1] = 'mean_region'
    train_df = pd.merge(train_df,region1, on=['region_county'], how='left')
    propCounterCls.region1 = region1

    fips_cat = train_df.groupby(["fips"],as_index=False)["logerror"].aggregate("mean")
    fips1=pd.DataFrame(fips_cat)
    fips1.columns.values[1] = 'mean_fips'
    train_df = pd.merge(train_df,fips1, on=['fips'], how='left')
    propCounterCls.fips1 = fips1

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    train_df = calculate_features(train_df, propCounterCls)

    train_df.zip_std = train_df.zip_std.fillna(  train_df.zip_std.median()  )
    train_df.city_std = train_df.city_std.fillna(  train_df.city_std.median()  )
    train_df.hood_std = train_df.hood_std.fillna(  train_df.hood_std.median()  )

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    train_df = train_df[ train_df.logerror < np.percentile(train_df.logerror,99) ]
    train_df = train_df[ train_df.logerror > np.percentile(train_df.logerror,1) ]
    train_df.reset_index(drop=True,inplace=True)




    #test_dates = ['2016-10-01','2016-11-01',
    #                '2016-12-01','2017-10-01',
    #                '2017-11-01','2017-12-01']
    #test_columns = ['201610','201611','201612','201710','201711','201712']

    print(datetime.today().strftime("%H:%M:%S %A %d. %B %Y"))
    #submission_df = pd.read_csv("sample_submission.csv")

    valuation_2017(train_df,prop_df,propCounterCls)



def main():
    process1()

if __name__ == "__main__":
    main()
