#
# import libariry
#

import numpy as np
import pandas as pd
# data precession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from trainClass import trainClass

from catboost import CatBoostRegressor
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import xgboost as  xgb
import random
import lightgbm as lgb
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#
# version 36 -> 6436
#
#version 29 -> LB:0.6446
#   add more feature
#
#version 28 -> LB:0.6445
#   model params 'n_estimators' -> 100
#
# version 26 -> LB:0.6443
#   model params 'n_estimators' -> 50
#
trainCls = trainClass()

def add_date_features(df):
    df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )

    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    #df.drop(["transactiondate"], inplace=True, axis=1)
    return df


def load_data(train,properties,datem):

    #train_2017 = trainCls.readTrain2017()   # pd.read_csv('../input/train_2017.csv')

    #train = pd.concat([train_2016, train_2017], ignore_index=True)
    #properties = trainCls.readProp2017()  #pd.read_csv('../input/properties_2017.csv')
    sample = trainCls.readSampleSub() #pd.read_csv('../input/sample_submission.csv')

    print("Preprocessing...")
    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)

    print("Set train/test data...")


    #
    # Add Feature
    #
    # life of property
    properties['N-life'] = 2018 - properties['yearbuilt']

    properties['A-calculatedfinishedsquarefeet'] = properties['finishedsquarefeet12'] + properties['finishedsquarefeet15']

    # error in calculation of the finished living area of home
    properties['N-LivingAreaError'] = properties['calculatedfinishedsquarefeet'] / properties['finishedsquarefeet12']

    # proportion of living area
    properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
    properties['N-LivingAreaProp2'] = properties['finishedsquarefeet12'] / properties['finishedsquarefeet15']

    # Amout of extra space
    properties['N-ExtraSpace'] = properties['lotsizesquarefeet'] - properties['calculatedfinishedsquarefeet']
    properties['N-ExtraSpace-2'] = properties['finishedsquarefeet15'] - properties['finishedsquarefeet12']

    # Total number of rooms
    properties['N-TotalRooms'] = properties['bathroomcnt'] + properties['bedroomcnt']

    # Average room size
    #properties['N-AvRoomSize'] = properties['calculatedfinishedsquarefeet'] / properties['roomcnt']

    # Number of Extra rooms
    properties['N-ExtraRooms'] = properties['roomcnt'] - properties['N-TotalRooms']

    # Ratio of the built structure value to land area
    properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    #properties['N-GarPoolAC'] = ((properties['garagecarcnt'] > 0) & (properties['pooltypeid10'] > 0) & (properties['airconditioningtypeid'] != 5)) * 1

    properties["N-location"] = properties["latitude"] + properties["longitude"]
    properties["N-location-2"] = properties["latitude"] * properties["longitude"]
    #properties["N-location-2round"] = properties["N-location-2"].round(-4)

    # Ratio of tax of property over parcel
    properties['N-ValueRatio'] = properties['taxvaluedollarcnt'] / properties['taxamount']

    # TotalTaxScore
    properties['N-TaxScore'] = properties['taxvaluedollarcnt'] * properties['taxamount']

    # polnomials of tax delinquency year
    properties["N-taxdelinquencyyear-2"] = properties["taxdelinquencyyear"] ** 2
    properties["N-taxdelinquencyyear-3"] = properties["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    properties['N-live'] = 2018 - properties['taxdelinquencyyear']

    # Number of properties in the zip
    zip_count = properties['regionidzip'].value_counts().to_dict()
    properties['N-zip_count'] = properties['regionidzip'].map(zip_count)

    # Number of properties in the city
    city_count = properties['regionidcity'].value_counts().to_dict()
    properties['N-city_count'] = properties['regionidcity'].map(city_count)

    # Number of properties in the city
    region_count = properties['regionidcounty'].value_counts().to_dict()
    properties['N-county_count'] = properties['regionidcounty'].map(region_count)

    id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
    for c in properties.columns:
        #if properties[c].dtype == 'object':
        #    lbl = LabelEncoder()
        #    lbl.fit(list(properties[c].values))
        #    properties[c] = lbl.transform(list(properties[c].values))
        if c in id_feature:
            properties[c]=properties[c].fillna(-999)
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            dum_df = pd.get_dummies(properties[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            properties = pd.concat([properties,dum_df],axis=1)
            properties = properties.drop([c], axis=1)
            #print np.get_dummies(properties[c])
    #
    # Make train and test dataframe
    #
    train = add_date_features(train)
    train = train.merge(properties, on='parcelid', how='left')
    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')

    missing_perc_thresh = 0.98
    exclude_missing6 = []
    num_rows = train.shape[0]
    for c in train.columns:
        num_missing = train[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing6.append(c)
    print(" We exclude: %s" % exclude_missing6)
    print(len(exclude_missing6))

    exclude_unique6 = []
    for c in train.columns:
        num_uniques = len(train[c].unique())
        if train[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1 and not "__nan__" in c:
            exclude_unique6.append(c)
    print("**  We exclude: %s" % exclude_unique6)
    print(len(exclude_unique6))

    exclude_other6 = ['parcelid', 'logerror']  # for indexing/training only
    # do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
    exclude_other6.append('propertyzoningdesc')
    exclude_other6.append('propertycountylandusecode')
    exclude_other6.append('transactiondate')

    train6_features = []
    for c in train.columns:
        if c not in exclude_missing6 \
           and c not in exclude_other6 and c not in exclude_unique6:
            train6_features.append(c)
    print("We use these for training: %s" % train6_features)
    print(len(train6_features))

    cat_feature_inds6 = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train6_features):
        num_uniques = len(train[c].unique())
        if num_uniques < cat_unique_thresh \
           and not 'sqft' in c \
           and not 'cnt' in c \
           and not 'nbr' in c \
           and not 'std' in c \
           and not 'N' in c \
           and not 'mean' in c \
           and not 'med' in c and not 'cos' in c and not 'sin' in c and c != "znull" \
           and not 'number' in c and not "__nan__" in c:
            cat_feature_inds6.append(i)

    print("** Cat features are: %s" % [train6_features[ind] for ind in cat_feature_inds6])


    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)

    # drop out ouliers
    #train = train[train.logerror > -0.4]
    #train = train[train.logerror < 0.418]
    x_train = train[train6_features]
    #x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = train["logerror"].values

    print("2016 date :",  datem)
    test = test.assign( transactiondate = datem)
    test = add_date_features(test)

    x_test = test[x_train.columns]

    del test, train
    print(x_train.shape, y_train.shape, x_test.shape)

    return x_train, y_train, x_test, cat_feature_inds6

def main():

    train = trainCls.readTrain2016()   #pd.read_csv('../input/train_2016_v2.csv')
    properties = trainCls.readProp2016()  #pd.read_csv('../input/properties_2017.csv')
    x_train, y_train, x_test, cat_feature_inds = load_data(train,properties,"2016-10-01")
    #return

    num_ensembles = 5
    y_pred = 0.0
    print("* start to train 2016 data....")
    for i in tqdm(range(num_ensembles)):
        # TODO(you): Use CV, tune hyperparameters
        #model = CatBoostRegressor(
        #    iterations=200, learning_rate=0.03,
        #    depth=6, l2_leaf_reg=3,
        #    loss_function='MAE',
        #    eval_metric='MAE',
        #    random_seed=i)

        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)

        model.fit(
            x_train, y_train,
            cat_features=cat_feature_inds)
        y_pred += model.predict(x_test)

    y_pred /= num_ensembles
    df6 = pd.DataFrame(y_pred,columns=["201610"])
    df6.to_csv("output/y_pred2016_N-Feat.csv",index=False)

    train = trainCls.readTrain2017()   #pd.read_csv('../input/train_2016_v2.csv')
    properties = trainCls.readProp2017()  #pd.read_csv('../input/properties_2017.csv')
    x_train, y_train, x_test, cat_feature_inds = load_data(train,properties,"2017-10-01")

    num_ensembles = 5
    y_pred = 0.0
    print("* start to train 2017 data....")
    for i in tqdm(range(num_ensembles)):
        # TODO(you): Use CV, tune hyperparameters
        #model = CatBoostRegressor(
        #    iterations=200, learning_rate=0.03,
        #    depth=6, l2_leaf_reg=3,
        #    loss_function='MAE',
        #    eval_metric='MAE',
        #    random_seed=i)

        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)

        model.fit(
            x_train, y_train,
            cat_features=cat_feature_inds)
        y_pred += model.predict(x_test)

    y_pred /= num_ensembles
    df7 = pd.DataFrame(y_pred,columns=["201710"])
    df7.to_csv("output/y_pred2017_N-Feat.csv",index=False)


if __name__ == "__main__":
    main()
