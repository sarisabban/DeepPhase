import os
import h5py
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn import utils
from keras import Input, Model
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization, Lambda, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, concatenate, Reshape
def warn(*args, **kwargs): pass
warnings.warn = warn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with h5py.File('Y.hdf5', 'r') as Yh:
	Y = Yh['default'][()]
with h5py.File('Space.hdf5', 'r') as Sh:
	Space = Sh['default'][()]
with h5py.File('UnitC.hdf5', 'r') as Uh:
	UnitC = Uh['default'][()]
with h5py.File('Coord.hdf5', 'r') as Ch:
	Coord = Ch['default'][()]
# Split train/tests/valid
Xs_train, Xs_tests, Xu_train, Xu_tests, Xc_train, Xc_tests, Y_train, Y_tests =\
	train_test_split(Space, UnitC, Coord, Y, test_size=0.20)
Xs_train, Xs_valid, Xu_train, Xu_valid, Xc_train, Xc_valid, Y_train, Y_valid =\
	train_test_split(Xs_train, Xu_train, Xc_train, Y_train, test_size=0.25)
''' Neural Network '''
shapeC = Coord.shape[1:]
n_feat = shapeC[0]
n_chan = shapeC[1]
n_chan2= n_chan*n_chan
n_clas = 2
A = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
def mat_mul(A, B): return tf.matmul(A, B)
input_points = Input(shape=(n_feat, n_chan))
x = Conv1D(64, 1, activation='relu', input_shape=(n_feat, n_chan))(input_points)
x = BatchNormalization()(x)
x = Conv1D(128, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(1024, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool1D(pool_size=n_feat)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(n_chan2, weights=[np.zeros([256, n_chan2]), np.array(A).astype(np.float32)])(x)
input_T = Reshape((n_chan, n_chan))(x)
g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = Conv1D(64, 1, input_shape=(n_feat, n_chan), activation='relu')(g)
g = BatchNormalization()(g)
g = Conv1D(64, 1, input_shape=(n_feat, n_chan), activation='relu')(g)
g = BatchNormalization()(g)
f = Conv1D(64, 1, activation='relu')(g)
f = BatchNormalization()(f)
f = Conv1D(128, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = Conv1D(1024, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = MaxPool1D(pool_size=n_feat)(f)
f = Dense(512, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(256, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(4096, weights=[np.zeros([256, 4096]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)
g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = Conv1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Conv1D(128, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Conv1D(1024, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = MaxPool1D(pool_size=n_feat)(g)
c = Dense(512, activation='relu')(g)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(256, activation='relu')(c)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(n_clas, activation='softmax')(c)
c = Flatten()(c)
modelC = Model(inputs=input_points, outputs=c)
modelC.compile(optimizer=Adam(lr=0.001, decay=0.7), loss='categorical_crossentropy', metrics=['accuracy'])
modelC.fit(Xc_train, Y_train, validation_data=(Xc_valid, Y_valid), epochs=200, batch_size=32)
