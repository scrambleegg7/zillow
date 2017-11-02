#
from trainClass import trainClass

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn import model_selection, preprocessing
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier

from time import time

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

# use edward for posterior

import tensorflow as tf
from edward.models import Normal
import edward as ed

# target columns name
target = "logerror"

def analysis01(trainCls):

    data_df = trainCls.merge1()
    print("-"*30)
    print("merged shape...")
    print(data_df.shape)

    # top3 effect columns excl latitute / longitude
    fixed_effect_predictions = [
        "lotsizesquarefeet",
        "structuretaxvaluedollarcnt",
        "landtaxvaluedollarcnt",
        "logerror"
    ]

    #drop na
    data_df = data_df[ fixed_effect_predictions ].dropna()

    #data_df["regionidzip"] = data_df["regionidzip"].astype('category').cat.codes
    #zip_codes = np.array(zip_codes)
    print("-"*30)
    print(data_df.head())

    # include random effect
    _X = data_df.ix[:,:2].values
    _y = data_df.ix[:,-1].values

    #split train / val
    msk = np.random.rand( _X.shape[0] ) < 0.5
    top_data = 39000
    X_train = _X[msk][:top_data]
    X_val = _X[~msk][:top_data]
    y_train = _y[msk][:top_data]
    y_val = _y[~msk][:top_data]
    print("-"*30)
    print("** train shape ..", X_train.shape,y_train.shape)
    print("** valid shape",X_val.shape,y_val.shape)

#    select_df =


    return X_train,X_val,y_train,y_val

def evaluate(X_train,X_val,y_train,y_val):

    N,D = X_train.shape

    fixed_effects = tf.placeholder(tf.float32,[N,D])

    # N(0,1)
    beta_fixed_effects = Normal(loc=tf.zeros(D), scale=tf.ones(D))
    alpha = Normal(loc=tf.zeros(1),scale=tf.ones(1))

    mu_y = alpha + ed.dot(fixed_effects,beta_fixed_effects)
    y = Normal(loc=mu_y,scale=tf.ones(N))

    #qw
    q_beta_fixed_effects = Normal(
        loc=tf.Variable(tf.random_normal([D])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([D])))
    )
    #qb
    q_alpha = Normal(
        loc=tf.Variable(tf.random_normal([1])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([1])))
    )


    latent_vars = {
        beta_fixed_effects: q_beta_fixed_effects,
        alpha: q_alpha
    }

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())

    sess.run(init_op)
    inference = ed.KLqp(latent_vars, data={fixed_effects: X_train, y: y_train})
    inference.run(n_samples=5, n_iter=250)

    y_post = ed.copy(y,latent_vars)
    #y_post = Normal( loc=ed.dot(X_train ,q_beta_fixed_effects) + q_alpha, scale=tf.ones(N) )
    print(y_post.shape)

    print("MSE on test data ...")
    print( ed.evaluate("mean_squared_error", data={fixed_effects:X_val,y_post:y_val}) )

    print("Mean Absolute Error on test data ...")
    print( ed.evaluate("mean_absolute_error", data={fixed_effects:X_val,y_post:y_val}) )

    param_posteriors = {
        beta_fixed_effects: q_beta_fixed_effects.mean(),
        alpha: q_alpha.mean()
    }
    X_val_feed_dict = {fixed_effects:X_val}
    y_posterior = ed.copy(y,param_posteriors)

    print("* Mean Absolute Error on test data ...")
    print("* Mean validation %.6f" % y_val.mean() )
    print( ed.evaluate("mean_absolute_error", data={fixed_effects:X_val,y_posterior:y_val}) )



def cross_check(y_post,y_):

    pass

def main():
    trainCls = trainClass(test=True)
    X_train,X_val,y_train,y_val = analysis01(trainCls)

    y_post = evaluate(X_train,X_val,y_train,y_val)



if __name__ == "__main__":
    main()
