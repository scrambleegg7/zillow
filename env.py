#
# zillow Data path
#

import numpy as np
import os

from os.path import join

def setEnv():

    envs = {}
    envs["data_dir"] = "/Users/donchan/Documents/myData/KaggleData/zillow"
    envs["train2016"] = join(envs["data_dir"],"train_2016_v2.csv")
    envs["prop"] = join(envs["data_dir"],"properties_2016.csv")

    envs["train2017"] = join(envs["data_dir"],"train_2017.csv")
    envs["prop2017"] = join(envs["data_dir"],"properties_2017.csv")


    envs["sample_submission"] = join(envs["data_dir"],"sample_submission.csv")

    envs["orig_kaggle"] = join(envs["data_dir"],"sub_kaggle_orginal20170807_141212_06437.csv")

    return envs
