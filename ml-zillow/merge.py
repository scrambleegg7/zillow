import pandas as pd

print 'Reading...'
train_df = pd.read_csv('data/train_2016.csv', parse_dates=['transactiondate'])
properties_df = pd.read_csv('data/properties_2016.csv')

print 'Merging...'
train_merge_df = pd.merge(train_df, properties_df, on='parcelid', how='left')

print 'Writing...'
train_merge_df.to_csv('data/train_merged_2016.csv')

