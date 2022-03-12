import math
import keras
import random
import statistics
import numpy as np
import tensorflow as tf
from keras import layers
from multiprocessing import Pool
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import pickle

class DataGenerator(keras.utils.Sequence):
	def __init__(self, X, Y, batch_size, feature_size):
		''' Initialization '''
		self.X = X
		self.Y = Y
		self.feature_size = feature_size
		self.batch_size = batch_size
		self.on_epoch_end()
	def on_epoch_end(self):
		''' Shuffle at end of epoch '''
		self.example_indexes = np.arange(len(self.X))
		number_of_batches = len(self.example_indexes)/self.batch_size
		self.number_of_batches = int(np.floor(number_of_batches))
		np.random.shuffle(self.example_indexes)
	def __len__(self):
		''' Denotes the number of batches per epoch '''
		return(int(np.floor(len(self.X)/self.batch_size)))
	def __getitem__(self, index):
		''' Generate one batch of data '''
		batch_indexes = self.example_indexes[index*self.batch_size:\
			(index+1)*self.batch_size]
		batch_x = np.array([self.X[k] for k in batch_indexes])
		batch_y = np.array([self.Y[k] for k in batch_indexes])
		x = []
		for example in batch_x:
			example = example[~np.isnan(example).any(axis=1)]
			idx = np.random.choice(len(example),\
				size=self.feature_size, replace=False)
			example = example[idx, :]
			x.append(example)
		batch_x = np.array(x)
		return batch_x, batch_y

NUM_CLASSES = 2
NUM_POINTS  = 300
CHANNELS    = 5

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)

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

def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer='zeros',
        bias_initializer=bias,
        activity_regularizer=reg)(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, CHANNELS))
x = tnet(inputs, CHANNELS)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['sparse_categorical_accuracy'])

######################################################################

print('[+] Testing XYZRE | Generator | 300 points | batch 32')

with h5py.File('X_train.h5', 'r') as xr: x_train = xr['default'][()]
with h5py.File('Y_train.h5', 'r') as yr: y_train = yr['default'][()]
with h5py.File('X_valid.h5', 'r') as xv: x_valid = xv['default'][()]
with h5py.File('Y_valid.h5', 'r') as yv: y_valid = yv['default'][()]
with h5py.File('X_tests.h5', 'r') as xt: x_tests = xt['default'][()]
with h5py.File('Y_tests.h5', 'r') as yt: y_tests = yt['default'][()]

train = DataGenerator(x_train, y_train, 32, 300)
valid = DataGenerator(x_valid, y_valid, 32, 300)
tests = DataGenerator(x_tests, y_tests, 32, 300)

model.fit(generator=train, validation_data=valid, epochs=200, verbose=2)

model.save_weights('weights.h5')
