import tensorflow as tf
import pandas as pd

import models

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016_total.csv', parse_dates=['transactiondate'])
models.fillna_df(eval_df)

eval_df = models.add_outlier_column(eval_df)
eval_df = models.add_sign_column(eval_df)

model = models.logsign_classifier
results = model.evaluate(input_fn=lambda: models.input_fn(eval_df, 'logsign'), steps=1)
# results = model.evaluate(input_fn=lambda: models.input_fn(eval_df_outl), steps=1)
# results2 = model.evaluate(input_fn=lambda: models.input_fn(eval_df), steps=1)

print 'Results:'
print results
print model.get_variable_names()
# models.print_dnn(model)
# print 'Logits:'
# for weight in model.get_variable_value('dnn/logits/weights').flatten():
#     print '  {: .3f}x + {: .3f}'.format(weight, model.get_variable_value('dnn/logits/biases')[0])
# print model.get_variable_value('dnn/hiddenlayer_0/weights')
# print model.get_variable_value('dnn/hiddenlayer_0/biases')

input_samples = eval_df.sample(n=20)

output_samples = list(model.predict(input_fn=lambda: models.input_fn(input_samples, 'logsign'),
                                    outputs=None))
print 'Selected Results'
print '{:<20}{:<20}{:<20}{:<20}{:<20}'.format('totalinfo', 'taxamount', 'yearbuilt', 'prediction', 'actual')
for k in range(len(output_samples)):
    print '{:<20f}{:<20f}{:<20f}{:<20f}{:<20f}'.format(input_samples['totalinfo'].values[k],
                                                       input_samples['taxamount'].values[k],
                                                       input_samples['yearbuilt'].values[k],
                                                       output_samples[k],
                                                       input_samples['logerror'].values[k])
