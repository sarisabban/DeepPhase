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
from tensorflow.compat.v2.keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, concatenate, Reshape
def warn(*args, **kwargs): pass
warnings.warn = warn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GPUs = 8
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
tf.compat.v1.disable_eager_execution()
with tf.device('/cpu:0'):
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
	g = Flatten()(g)
	modelC = Model(input_points, g)
	# Unit cell network
	shapeU = UnitC.shape[1:]
	inputU = Input(shape=shapeU)
	modelU = Dense(1024, activation="relu")(inputU)
	modelU = Model(inputU, modelU)
	# Space group network
	shapeS = Space.shape[1:]
	inputS = Input(shape=shapeS)
	modelS = Dense(1024, activation="relu")(inputS)
	modelS = Model(inputS, modelS)
	# Combined
	combined = concatenate([modelU.output, modelS.output, modelC.output])
	model = Dense(512, activation='relu')(combined)
	model = BatchNormalization()(model)
	model = Dropout(rate=0.7)(model)
	model = Dense(256, activation='relu')(model)
	model = BatchNormalization()(model)
	model = Dropout(rate=0.7)(model)
	model = Dense(n_clas, activation='softmax')(model)
	model = Model([modelU.input, modelS.input, modelC.input], model)
model = multi_gpu_model(model, gpus=GPUs)
model.compile(	optimizer=Adam(lr=0.001, decay=0.7),
				loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(	x=[Xu_train, Xs_train, Xc_train], y=Y_train,
			validation_data=([Xu_valid, Xs_valid, Xc_valid], Y_valid),
			epochs=200, batch_size=32*GPUs, verbose=1)
predics= model.predict([Xu_tests, Xs_tests, Xc_tests], verbose=1)
y_preds= np.argmax(predics, axis=1)
y_tests= np.argmax(Y_tests, axis=1)
matrix = confusion_matrix(y_tests, y_preds)
print('-'*55)
print(classification_report(y_tests, y_preds, target_names=classes))
print('Confusion Matrix:\n{}'.format(matrix))
model.save_weights('DeepClass.h5')
