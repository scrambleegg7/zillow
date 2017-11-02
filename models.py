#
from env import setEnv
import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class modelClass(object):

    def __init__(self,test=False):

        self.envs = setEnv()
        self.test = test

        #self.setModelParams()


    def randomForestModel(self):

        estimator = RandomForestRegressor(random_state=0, n_estimators=300,n_jobs=-1)
        return estimator
        #score = cross_val_score(estimator, X_full, y_full).mean()

    def setModelParams(self):

        self.params_list = {}
        self.params_list["lightgbm1"] = self.setLightGMBParam()
        #self.params_list["xgb1"] = self.setXgboostParam1()

    def setXGBOOSTModel(self,y_hat):
        self.params_list = {}
        self.params_list["xgb1"] = self.setXgboostParam1(y_hat)
        self.params_list["xgb2"] = self.setXgboostParam2(y_hat)

        return self.params_list

    def getModelParams(self):

        return self.params_list

    def linearRegressionModel(self):
        reg = LinearRegression(n_jobs=-1)
        return reg

    def setLightGMBParam(self):
        print("\nSetting up data for LighGBM ...")
        params = {}
        params['max_bin'] = 18
        params['learning_rate'] = 0.100 # shrinkage_rate
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression'
        params['metric'] = 'l1'          # or 'mae'
        params['sub_feature'] = 0.596      # feature_fraction -- OK, back to .5, but maybe later increase this
        params['bagging_fraction'] = 0.79 # sub_row
        params['bagging_freq'] = 45
        params['num_leaves'] = 512        # num_leaf
        params['min_data'] = 700         # min_data_in_leaf
        params['min_hessian'] = 0.37     # min_sum_hessian_in_leaf
        params['verbose'] = 0
        params['feature_fraction_seed'] = 2
        params['bagging_seed'] = 3
        #params = {}
        #params['max_bin'] = 10
        #params['learning_rate'] = 0.0021 # shrinkage_rate
        #params['boosting_type'] = 'gbdt'
        #params['objective'] = 'regression'
        #params['metric'] = 'l1'          # or 'mae'
        #params['sub_feature'] = 0.3      # feature_fraction -- OK, back to .5, but maybe later increase this

        #params['bagging_fraction'] = 0.85 # sub_row
        #params['bagging_freq'] = 40
        #params['num_leaves'] = 512        # num_leaf
        #params['min_data'] = 500         # min_data_in_leaf
        #params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
        #params['verbose'] = 0
        #params['feature_fraction_seed'] = 2
        #params['bagging_seed'] = 3

        return params

    def setXgboostParam1(self):

        print("\n")
        print("-"*40)
        print("\nSetting up data for XGBoost 1 ...")
        # xgboost params
        params = {
            'eta': 0.038,
            #'max_depth': 5,
            'subsample': 0.85,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'colsample_bytree':0.80,
            'lambda': 0.8,
            'alpha': 0.4,
            'silent': 1
        }
        num_rounds = 1000
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def setXgboostParam2(self):

        print("\n")
        print("-"*40)
        print("\nSetting up data for XGBoost 2 ...")
        # xgboost params
        params = {
            'eta': 0.033,
            #'max_depth': 6,
            'subsample': 0.50,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.9,
            'alpha': 0.5,
            #'base_score': y_mean,
            'silent': 1
        }
        num_rounds = 1000
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def setXgboostParam3(self):

        print("\n")
        print("-"*40)
        print("\nSetting up data for XGBoost 3 ...")
        # xgboost params
        params = {
            'eta': 0.033,
            'max_depth': 6,
            'subsample': 0.4,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 1.8,
            'alpha': 1.6,
            #'base_score': y_mean,
            'silent': 1
        }
        num_rounds = 1000
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xgb01(self):
        params = {'objective': 'reg:linear',
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 6,
              'min_child_weight': 6,
              'colsample_bytree': 0.7,
              'eval_metric': 'mae',
              'silent': 1
        }
        num_rounds = 1000
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel1(self):

        params = {'objective': 'reg:linear',
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 9,
              'min_child_weight': 6,
              'colsample_bytree': 0.7,
              'silent': 1
              }
        num_rounds = 1000
        print('Model 1 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel2(self):
        params = {'objective': 'count:poisson',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }

        num_rounds = 5000
        print('Model 2 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel3(self):
        params = {'objective': 'count:poisson',
              'eta': 0.005,
              'subsample': 0.7,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 3 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel4(self):

        params = {'objective': 'reg:linear',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 6,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 4 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params


    def xModel5(self):
        # Build Model 5
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.7,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 5 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel6(self):
        # Build Model 6
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 6 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel7(self):
        # Build Model 7
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 7 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel8(self):

        # Build Model 8
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 8 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel9(self):
        # Build Model 9
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
        num_rounds = 5000
        print('Model 9 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params

    def xModel10(self):
        # Build Model 10
        params = {'objective': 'reg:linear',
              'eta': 0.003,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }

        num_rounds = 5000
        print('Model 9 has been built!')
        xgb_params=[list(params.items()), num_rounds]
        return xgb_params
