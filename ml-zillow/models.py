import tensorflow as tf
import math
import numpy as np

# --- Input ---

def input_fn(df, output_label='logerror'):
    columns = {
        k: tf.constant(df[k].values)
        for k in [col.name for col in feature_columns] if not k.endswith('_bucketized')
    }
    print 'variance:'
    print df[output_label].var()
    output = tf.constant(df[output_label].values, dtype=tf.float64)

    print 'columns, output'
    print columns, output
    return columns, output

def fillna_df(df):
    for k in df:
        if df[k].dtype.kind in 'iufc' and df[k].name != 'logerror':
            df[k].fillna(df[k].mean() if not math.isnan(df[k].mean()) else 0, inplace=True)
            df[k]=(df[k]-df[k].mean())/df[k].std()

def add_outlier_column(df):
    """Adds a new column that is zero-valued if the logerror of the
    row is within one standard deviation of the mean."""

    mean = df['logerror'].mean()
    std_deviation = df['logerror'].std()
    df['is_outlier'] = df.apply(lambda row: abs(row['logerror'] - mean) > std_deviation, axis=1)
    return df

def add_sign_column(df):
    """Adds a new column that is True if the logerror of the row is
    positive and False otherwise."""

    df['logsign'] = df.apply(lambda row: row['logerror'] >= 0, axis=1)
    return df

# --- Debugging ---

def _print_layer(weights, biases):
    bias = biases[0]
    if len(biases) != 1:
        print 'len biases: {:d} len weights: {:d}'.format(len(biases), len(weights.flatten()))

    print weights
    print biases

    # for j in range(len(weights[0])):
    #     for range(len(weights)):

    for weight in weights:
        print ' '.join(map(lambda w: '{: .3f}'.format(w), weight))
    # for weight in weights.flatten():
    #     print '  {: .3f}x + {: .3f}'.format(weight, bias)

def print_dnn(dnn):
    num_hidden_layers = len(filter(lambda name: name.startswith('dnn/hiddenlayer') and name.endswith('weights'),
                                   dnn.get_variable_names()))
    print num_hidden_layers

    for i in range(num_hidden_layers):
        weights_var = 'dnn/hiddenlayer_{:d}/weights'.format(i)
        biases_var = 'dnn/hiddenlayer_{:d}/biases'.format(i)
        print 'layer {:d}'.format(i)
        _print_layer(dnn.get_variable_value(weights_var),
                     dnn.get_variable_value(biases_var))

    print 'logits'
    _print_layer(dnn.get_variable_value('dnn/logits/weights'),
                 dnn.get_variable_value('dnn/logits/biases'))

# --- Model ---

feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('totalinfo', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('bedroomcnt', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('calculatedbathnbr', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('calculatedfinishedsquarefeet', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('fullbathcnt', dtype=tf.float32, dimension=1),
    #tf.contrib.layers.real_valued_column('2error', dtype=tf.float32, dimension=1), #test feature: x=2/logerror
    tf.contrib.layers.real_valued_column('basementsqft', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('finishedsquarefeet12', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('finishedsquarefeet13', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('yardbuildingsqft26', dtype=tf.float32, dimension=1),


]
#feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[1], boundaries=range(0, 10)))
dnn_regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                              model_dir = './dnn-regressor-model_outl',
                                              hidden_units = [256,256,256,256],
                                              activation_fn = tf.nn.relu,
                                              dropout = .5,
                                              enable_centered_bias = True,
                                              label_dimension = 1,
                                              optimizer= tf.train.AdadeltaOptimizer(
                                              learning_rate=1,
                                              rho=0.99
                                              )
                                              )

outlier_classifier = tf.contrib.learn.DNNClassifier(hidden_units=[128, 128, 128],
                                                    feature_columns=feature_columns,
                                                    model_dir='./dnn-outlier-classifier')

logsign_classifier = tf.contrib.learn.DNNClassifier(hidden_units=[128, 128, 128],
                                                    feature_columns=feature_columns,
                                                    model_dir='./dnn-logsign-classifier')
