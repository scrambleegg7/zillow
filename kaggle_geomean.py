from __future__ import division
from collections import defaultdict
from glob import glob
import sys
import math
import os
import numpy as np

#glob_files = sys.argv[1]
#loc_outfile = sys.argv[2]

def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform",type="average"):
    if method == "average":
        scores = defaultdict(float)
    with open(loc_outfile,"wb") as outfile:
        for i, glob_file in enumerate( glob( glob_files ) ):
            print "parsing:", glob_file
            # sort glob_file by first column, ignoring the first line
            lines = open( glob_file   ).readlines()
            print(len(lines))
            print(lines[1])
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate( lines ):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    if type == "geo":
                        scores[(e,row[0])] *= float(row[1])
                    else:
                        scores[(e,row[0])] += float(row[1])
                        #print(scores[(e,row[0])])

        for j,k in sorted(scores):
            #print(scores[(j,k)], i)
            if type == "geo":
                outfile.write("%s,%.4f\n"%(k,np.power(scores[(j,k)],1/(i+1))))
            else:
                avg = scores[(j,k)] / ( i + 1 )
                outfile.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(k,avg,avg,avg,avg,avg,avg))

        print("wrote to %s"%loc_outfile)


def main():

    mydir = "/Users/donchan/Documents/Statistical_Mechanics/tensorflow/zillow/output/consolidated"
    glob_files = ["sub_XGBM_20170818_170406","sub_XGBM_20170818_173201",
                  "sub_XGBM_20170818_174700","sub_XGBM_20170818_165913"]

    loc_outfile = "output/geomean_out.csv"

    glob_files = mydir + "/*.csv"
    kaggle_bag(glob_files, loc_outfile)



if __name__ == "__main__":
    main()
