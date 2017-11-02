import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

from trainClass import trainClass

MAKE_SUBMISSION = True          # Generate output file.
CV_ONLY = False                 # Do validation only; do not generate predicitons.
FIT_FULL_TRAIN_SET = True       # Fit model to full training set after doing validation.
FIT_2017_TRAIN_SET = False      # Use 2017 training data for full fit (no leak correction)
FIT_COMBINED_TRAIN_SET = True   # Fit combined 2016-2017 training set
USE_SEASONAL_FEATURES = True
VAL_SPLIT_DATE = '2016-09-15'   # Cutoff date for validation split
LEARNING_RATE = 0.007           # shrinkage rate for boosting roudns
ROUNDS_PER_ETA = 20             # maximum number of boosting rounds times learning rate
OPTIMIZE_FUDGE_FACTOR = False   # Optimize factor by which to multiply predictions.
FUDGE_FACTOR_SCALEDOWN = 0.3    # exponent to reduce optimized fudge factor for prediction

def changeGeoData(_df):

    properties = _df.copy()

    print("\n")
    print("-"*40)
    print("\n      geo data divide 1e6")

    properties["latitude"] = properties["latitude"].apply(lambda x:x / 1e6)
    properties["longitude"] = properties["longitude"].apply(lambda x:x / 1e6)

    #properties["over_under_range_latitude"] = properties["latitude"].apply(lambda x:1 if x >= 33.8 and x <= 34.15 else 0 )
    #properties["over_under_range_longitude"] = properties["longitude"].apply(lambda x:1 if x >= -118.5 and x <= -118.25 else 0 )

    del _df
    return properties

def setFeaturesSumAndVar(_df):

    properties=_df.copy()
    print("\n")
    print("-"*30)
    print("\n    set Sum/Var of Features of properties ...")
    print("-"*30)
    properties["features_sum"] = properties.sum(axis=1).values.reshape(-1,1)
    #properties["features_var"] = properties.var(axis=1).values.reshape(-1,1)

    scaler = StandardScaler()
    target_list = ["features_sum","features_var"]
    #for c in target_list:
    properties["features_sum"] = scaler.fit_transform(properties["features_sum"].values.reshape(-1,1))

    del _df
    return properties

# similar to the1owl
def add_date_features(df):
    df = df.assign( transactiondate = lambda x: pd.to_datetime(x.transactiondate)   )

    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    #df.drop(["transactiondate"], inplace=True, axis=1)
    return df

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
    del _df
    #self.setProp(properties)
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

    #print(properties["taxamount"].describe().transpose() )

                #print("%s %s" % (col, properties[col].dtype))
    del _df
    return properties


def main():

    trainCls = trainClass()
    train7_df = trainCls.readTrain2017()
    #pd.read_csv('./train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

    test_df = trainCls.readSampleSub()
    #pd.read_csv('./sample_submission.csv', low_memory=False)
    properties_7 = trainCls.readProp2017()
    test_df['parcelid'] = test_df['ParcelId']

    print("2017 Training Size:" + str(train7_df.shape))
    print("2017 Property Size:" + str(properties_7.shape))

    properties_7 = changeGeoData(properties_7)
    properties_7 = setFeaturesSumAndVar(properties_7)

    for c in properties_7.columns.tolist():
        properties_7[c + "__nan__"] = properties_7[c].apply(lambda x:1 if pd.isnull(x) else 0)

    numberOfNullCols7 = properties_7.isnull().sum(axis=1)
    properties_7 = changeObjects(properties_7)

    properties_7 = changeDataTypes(properties_7)

    d_median7 = properties_7.median(axis=0)

    properties_7["znull"] = numberOfNullCols7

    train7_df = add_date_features(train7_df)
    train7_df = train7_df.merge(properties_7, how='left', on='parcelid')

    test7_df = test_df.merge(properties_7, how='left', on='parcelid')

    print("Train: ", train7_df.shape)
    print("Test: ", test7_df.shape)

    missing_perc_thresh = 0.98
    exclude_missing7 = []
    num_rows = train7_df.shape[0]
    for c in train7_df.columns:
        num_missing = train7_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing7.append(c)
    print("2017 We exclude: %s" % exclude_missing7)
    print(len(exclude_missing7))

    exclude_unique7 = []
    for c in train7_df.columns:
        num_uniques = len(train7_df[c].unique())
        if train7_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1 and not "__nan__" in c:
            exclude_unique7.append(c)
    print("** 2017 We exclude: %s" % exclude_unique7)
    print(len(exclude_unique7))

    exclude_other7 = ['parcelid', 'logerror']  # for indexing/training only
    # do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
    exclude_other7.append('propertyzoningdesc')
    exclude_other7.append('transactiondate')

    train7_features = []
    for c in train7_df.columns:
        if c not in exclude_missing7 \
           and c not in exclude_other7 and c not in exclude_unique7:
            train7_features.append(c)
    print("2017 We use these for training: %s" % train7_features)
    print(len(train7_features))

    cat_feature_inds7 = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train7_features):
        num_uniques = len(train7_df[c].unique())
        if num_uniques < cat_unique_thresh \
           and not 'sqft' in c \
           and not 'cnt' in c \
           and not 'nbr' in c \
           and not 'std' in c and not 'N' in c  \
           and not 'mean' in c and not 'med' in c and c != "znull" \
           and not 'number' in c and not "__nan__" in c:
            cat_feature_inds7.append(i)

    print("** 2017 Cat features are: %s" % [train7_features[ind] for ind in cat_feature_inds7])

    train7_df = train7_df.fillna(d_median7)
    test7_df = test7_df.fillna(d_median7)

    X_train7 = train7_df[train7_features]
    y_train7 = train7_df.logerror
    print(X_train7.shape, y_train7.shape)

    dates2017 = ["2017-10-01", "2017-11-01", "2017-12-01"]
    y_pred2017 = np.zeros(( test7_df.shape[0],3 ))
    for m in range(3):
        print("2017 date", dates2017[m])
        test7_df = test7_df.assign( transactiondate = dates2017[m])
        test7_df = add_date_features(test7_df)
        X_test7 = test7_df[train7_features]

        num_ensembles = 5
        y_pred7 = 0.0
        for i in tqdm(range(num_ensembles)):
            # TODO(you): Use CV, tune hyperparameters
            model = CatBoostRegressor(
                iterations=200, learning_rate=0.03,
                depth=6, l2_leaf_reg=3,
                loss_function='MAE',
                eval_metric='MAE',
                random_seed=i)
            model.fit(
                X_train7, y_train7,
                cat_features=cat_feature_inds7)
            y_pred7 += model.predict(X_test7)
        y_pred7 /= num_ensembles
        y_pred2017[:,m] = y_pred7

    df7 = pd.DataFrame(y_pred2017,columns=["201710","201711","201712"])
    df7.to_csv("output/y_pred2017_nan_.csv",index=False)
    print(df7.head())

if __name__ == "__main__":
    main()
