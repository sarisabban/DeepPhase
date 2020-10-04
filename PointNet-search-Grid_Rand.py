import h5py
import keras
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

with h5py.File('dataset64/Y.hdf5'    , 'r') as yh: Y     = yh['default'][()]
with h5py.File('dataset64/Coord.hdf5', 'r') as ch: Coord = ch['default'][()]

X_train, X_tests, Y_train, Y_tests = train_test_split(Coord,   Y,       test_size=0.20)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25)

NUM_POINTS  = 10000
NUM_CLASSES = 2
CHANNELS    = X_train.shape[-1]

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def create_model(lr, drop):
    inputs = keras.Input(shape=(NUM_POINTS, CHANNELS))
    bias = keras.initializers.Constant(np.eye(CHANNELS).flatten())
    reg = OrthogonalRegularizer(CHANNELS)
    y = layers.Dense(32)(inputs)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(64)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(512)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.GlobalMaxPooling1D()(y)
    y = layers.Dense(256)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(128)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(
        CHANNELS * CHANNELS,
        kernel_initializer='zeros',
        bias_initializer=bias,
        activity_regularizer=reg)(y)
    feat_T = layers.Reshape((CHANNELS, CHANNELS))(y)
    x = layers.Dot(axes=(2, 1))([inputs, feat_T])
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    bias = keras.initializers.Constant(np.eye(32).flatten())
    reg = OrthogonalRegularizer(32)
    y = layers.Dense(32)(x)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(64)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(512)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.GlobalMaxPooling1D()(y)
    y = layers.Dense(256)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(128)(y)
    y = layers.BatchNormalization(momentum=0.0)(y)
    y = layers.Activation('relu')(y)
    y = layers.Dense(
        32 * 32,
        kernel_initializer='zeros',
        bias_initializer=bias,
        activity_regularizer=reg)(y)
    feat_T = layers.Reshape((32, 32))(y)
    x = layers.Dot(axes=(2, 1))([x, feat_T])
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=['sparse_categorical_accuracy'])
    return(model)

def Train(lr=0.001, drop=0.3):
    model = create_model(lr, drop)
    model.fit(X_train, Y_train,
        epochs=100,
        validation_data=(X_valid, Y_valid),
        batch_size=32,
        verbose=0)
    evaluation = model.evaluate(X_tests, Y_tests, verbose=0)
    return(evaluation)

def Grid():
	param = {	'lr'   : [0.001],
				'drop' : [0.3]}
	keys, values = zip(*param.items())
	count = 1
	for item in values: count *= len(item)
	experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
	for i in enumerate(experiments):
		loss, acc = Train(**i[1])
		keras.backend.clear_session()
		t = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		print('{}\t {} loss {:5,} - accuracy {:5,}'.\
        format(t, i[1], round(loss, 5), round(acc, 5)))
	print('----\nDone')

def Rand():
	param = {	'lr'   : (1e-3, 1e-5),
				'drop' : (0.25, 0.50)}
	iters = 20
	for i in range(1, iters+1):
		experiment = {k: np.random.uniform(*v) for k, v in param.items()}
		loss, acc  = Train(**experiment)
		keras.backend.clear_session()
		t = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		print('{}\t {} loss {:5,} - accuracy {:5,}'.\
        format(t, experiment, round(loss, 5), round(acc, 5)))
	print('----\nDone')

#Grid()
Rand()
