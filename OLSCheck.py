# Parameters
# XGB_WEIGHT = 0.6840
XGB_WEIGHT = 0.63
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0550

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

#
# set XGB_WEIGHT = 0.63
#

# version 42
#    Going to try putting subsequent version comments at the bottom,
#      hoping that will might solve the problem with LightGBM results
#        changing when comments are the only difference

# version 41
#    Revert to single XGB model, as in version 37

# version 40
#    Try a different average (rounds=240 and 250) -> LB .0644095

# version 38
#    Try two XGB fits averaged (rounds=250 and 260) -> LB .0644081

# version 37
#    Shot in the dark: num_boost_rounds=250 for XGB -> LB .0644074

# version 36
#    Back to 0.6300, and set num_boost_rounds per CV -> LB .0644207

# version 34
#    Try XGB_WEIGHT = 0.6315 -> .0644087

# versomp 33
#    Tru XGB_WEIGHT = 0.6334, to do new quadratic approximation
#    LB .0644087

# version 32
#    Try XGB_WEIGHT = 0.63, LB = .0644086

# version 31
#    Change threshold to 0.419, per huiqin, LB = .0644096

# version 30
#    OK, never mind...

# version 28
#    Get rid of lot size, per Jayaraman

# version 27
#    Roll back to version 24/21

# version 26
#    Last one was wrong. I meant to bring back fireplacecnt.
#    LB 0.0644248, even worse than the mistake

# version 25
#    Bring back fireplaceflag, LB 0.0644245, never mind

# version 24
#    Revert to old BASELINE_WEIGHT and some cleanup

# version 21
#    Try BASELINE_WEIGHT = 0.0050, LB .0644125

# version 12
#    Try XGB_WEIGHT .6266, to do quadratic approximation, LB .0644123, already near-optimal

# version 11
#    Try XGB_WEIGHT .620 -> .6166, closer to old proportion, LB .0644133

# version 10
#    OLS_WEIGHT=.055, LB .0644127

# version 9
#    OLS_WEIGHT=.05, LB .0644129

# version 8
#    OLS_WEIGHT=.07, LB .0644136

# version 7
#    First attempt:
#    OLS_WEIGHT=.06, LB .0644129
#    (XGB_WEIGHT=.62 to keep proportion with LGB roughly same as in old script)




# THE FOLLOWING SERIES OF COMMENTS REFERS TO VERSIONS OF
#    https://www.kaggle.com/aharless/xgb-w-o-outliers-lgb-with-outliers-combined
#    from which this script was forked

# version 61
#   Drop fireplacecnt and fireplaceflag, following Jayaraman:
#     https://www.kaggle.com/valadi/xgb-w-o-outliers-lgb-with-outliers-combo-tune5

# version 60
#   Try BASELINE_PRED=0.0115, since that's the actual baseline from
#     https://www.kaggle.com/aharless/oleg-s-original-better-baseline

# version 59
#   Looks like 0.0056 is the optimum BASELINE_WEIGHT

# versions 57, 58
#   Playing with BASELINE_WEIGHT parameter:
#     3 values will determine quadratic approximation of optimum

# version 55
#   OK, it doesn't get the same result, but I also get a different result
#     if I fork the earlier version and run it again.
#   So something weird is going on (maybe software upgrade??)
#   I'm just going to submit this version and make it my new benchmark.

# version 53
#   Re-parameterize ensemble (should get same result).

# version 51
#   Quadratic approximation based on last 3 submissions gives 0.3533
#     as optimal lgb_weight.  To be slightly conservative,
#     I'm rounding down to 0.35

# version 50
#   Quadratic approximation based on last 3 submissions gives 0.3073
#     as optimal lgb_weight

# version 49
#   My latest quadratic approximation is concave, so I'm just taking
#     a shot in the dark with lgb_weight=.3

# version 45
#   Increase lgb_weight to 0.25 based on new quadratic approximation.
#   Based on scores for versions 41, 43, and 44, the optimum is 0.261
#     if I've done the calculations right.
#   I'm being conservative and only going 2/3 of the way there.
#   (FWIW my best guess is that even this will get a worse score,
#    but you gotta pay some attention to the math.)

# version 44
#   Increase lgb_weight to 0.23, per Nikunj's suggestion, even though
#     my quadratic approximation said I was already at the optimum

# verison 43
#   Higher lgb_weight, so I can do a quadratic approximation of the optimum

# version 42
#   The answer to the ultimate question of life, the universe, and everything
#     comes down to a slightly higher lgb_weight

# version 41
#   Trying Nikunj's suggestion of imputing missing values.

# version 39
#   Trying higher lgb_weight again but with old learning rate.
#   The new one did better with LGB only but makes the combination worse.

# version 38
#   OK back to baseline 0.2 weight

# version 37
#   Looks like increasing lgb_weight was better

# version 34
#   OK, try reducing lgb_weight instead

# version 32
#   Increase lgb_weight because LGB performance has imporved more than XGB
#   Increase learning rate for LGB: 0029 is compromise;  CV prefers 0033
#     (and reallly would prefer more boosting rounds with old value instead
#      but constaints on running time are getting hard)

# Version 27:
#   Control LightGBM's loquacity

# Version 26:
# Getting rid of the LightGBM validation, since this script doesn't use the result.
# Now use all training data to fit model.
# I have a separate script for validation:
#    https://www.kaggle.com/aharless/lightgbm-outliers-remaining-cv


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt

from sklearn import metrics


from trainClass import trainClass

##### READ IN RAW DATA

print( "\nReading data from disk ...")
#prop = pd.read_csv('../input/properties_2016.csv')
#train = pd.read_csv("../input/train_2016_v2.csv")

trainCls = trainClass()
train = trainCls.readTrain2016()
prop = trainCls.readProp2016()

################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols

np.random.seed(17)
random.seed(17)

#train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
#properties = pd.read_csv("../input/properties_2016.csv")
#submission = pd.read_csv("../input/sample_submission.csv")

train = trainCls.readTrain2016()
properties = trainCls.readProp2016()
submission = trainCls.makeSampleForTesting()

print(len(train),len(properties),len(submission))

def get_features(df):

    df.loc[:,"transactiondate"] = pd.to_datetime(df["transactiondate"])
    df.assign( transactiondate_year = lambda x: x.transactiondate.dt.year   )
    df.assign( transactiondate_month = lambda x: x.transactiondate.dt.year   )
    df["transactiondate"].apply(lambda x: x.transactiondate.dt.quarter   )


    #df.loc["transactiondate_year"] = df["transactiondate"].dt.year
    #df.loc["transactiondate_month"] = df["transactiondate"].dt.month
    #df.loc['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    loss = metrics.mean_absolute_error(y,y_pred)
    return loss
    #return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

print("exclude col names ..")
print(exc)
print("target col nam ..")
print(col)

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')

print("MAE .....for training data ...")
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']




########################
########################
##  Combine and Save  ##
########################
########################


##### COMBINE PREDICTIONS

#print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
#lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
#xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
#baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
#pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

#print( "\nCombined XGB/LGB/baseline predictions:" )
#print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    #pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    #submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i, test_dates[i])

#print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
#print( submission.head() )



##### WRITE THE RESULTS

#from datetime import datetime

#print( "\nWriting results to disk ..." )
#submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")




########################
########################
##  Version Comments  ##
########################
########################

#... version 42, continued from the top

# version 44
#    Confirmed that submitting the same exact script in immediate succession
#      produces the same result.

# version 45
#    Now try it with this comment added and see if it's still the same (and yes it is!).

# version 46
#    Add an extra XGB run with different parameters.  (LB score gets worse, but
#      perhaps because LightGBM results are different, even though I changed nothing
#        up to the point in the script where it finishes running LightGBM.  This is annoying.)

# version 47
#    Comment out version 46 changes.  But results will be completely new
#      because of black magic perpetrated by the LightGBM imps.

# version 54
#    Try higher XGB_WEIGHT (in this case .6900) per danieleewww -> LB .0644060

# version 56
#    Try XGB_WEIGHT=.71 -> LB .0644069

# version 57
#    Try XGB_WEIGHT=.67 -> LB .0644062
#      Quadratic approximation would imply optimum of about .684,
#         which I will note but not worth wasting another submission on it.
#      (Subsequent changes will alter the optimal weight anyhow.)

# version 58
#    Try bringing back the second XGB fit, but with a smaller weight (0.15 vs 0.40)
#      -> LB .0644053

# version 59
#    Try XGB_WEIGHT=.70 -> LB .0644056
#    Seems unlikely that this would reduce the LB score after the 2nd XGB model improved it.
#      I suspect that the LIghtGBM imps are up to their old tricks.

# version 60
#    Try XGB_WEIGHT=.684, per earlier approximated optimum
#    Also, I'm putting XGB1_WEIGHT at the top, although this will tempt
#       all sort of evil mischeif from the LightGBM imps
#    LB .0644052

# version 61
#    Try XGB1_WEIGHT=.80 -> LB .0644049

# version 62
#    Try XGB1_WEIGHT=.75 -> LB .0644055

# version 63
#    Change XGB1_WEIGHT to .8083 (quadratic approximated optimum)
#    Add ratio of taxvaluedollarcnt/taxamount (to LGB), per RpyGamer suggesiton
#    LB .0644238

# version 65
#    Get rid of tax ratio.  (Maybe try it later in XGB?)

# versions 66-70:  Nonconvexity or LightGBM imps?
#    66. Original OLS_WEIGHT  0.0550 -> LB .0644049
#    67. Reduce OLS_WEIGHT to 0.0500 -> LB .0644052
#    68. Riase OLS_WEIGHT  to 0.0650 -> LB .0644049
#    69. Reduce OLS_WEIGHT to 0.0600 -> LB .0655052
#    70. Reduce OLS_WEIGHT to 0.0550 again

# version 74: Try Amit's weights

# version 75: Revert to version 65 again

# version 77: Remove airconditioningtypeid, buildingclasstypeid, and buildingqualitytypeid
#               from XGB

# version 78: Put back airconditioningtypeid, buildingclasstypeid

# version 79: Revert to 65/70/75

# versino 80: Set some seeds for LightGBM, but I don't think they will chase away the imps
