import os
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
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, concatenate, Reshape

def warn(*args, **kwargs): pass
warnings.warn = warn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('DeepClass.csv')
data = data.fillna(0)
data = utils.shuffle(data)
''' X features '''
s = data[data.columns[2]].values
ce= data[data.columns[3:6]].values
ca= data[data.columns[6:9]].values
x = data[data.columns[9::5]].values
y = data[data.columns[10::5]].values
z = data[data.columns[11::5]].values
r = data[data.columns[12::5]].values
f = data[data.columns[13::5]].values
# One-hot encode s      [Space Groups]
categories = [sorted([x for x in range(1, 230+1)])]
s = s.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
# Normalise min/max ce  [Unit Cell Edges]
mini = np.amin(ce)
maxi = np.amax(ce)
ce = (ce-mini)/(maxi-mini)
# Normalise min/max ca  [Unit Cell Angles]
mini = 90.0
maxi = 180.0
ca = (ca-mini)/(maxi-mini)
# Normalise min/max x   [X Coordinates]
mini = -1
maxi = 1
x = (x-mini)/(maxi-mini)
# Normalise min/max y   [Y Coordinates]
mini = -1
maxi = 1
y = (y-mini)/(maxi-mini)
# Normalise min/max z   [Z Coordinates]
mini = -1
maxi = 1
z = (z-mini)/(maxi-mini)
# Normalise min/max r   [Resolution]
mini = 2.5
maxi = 10
r = (r-mini)/(maxi-mini)
# Final features
Space = onehot_encoder.fit_transform(s)
UnitC = np.concatenate([ce, ca], axis=1)
Coord = np.array([x, y, z, r, f])
Coord = np.swapaxes(Coord, 0, 2)
Coord = np.swapaxes(Coord, 0, 1)
''' Y labels '''
labels = data[data.columns[1]].values
classes = list(np.unique(labels))
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)
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
model.compile(	optimizer=Adam(lr=0.001, decay=0.7),
				loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(	x=[Xu_train, Xs_train, Xc_train], y=Y_train,
			validation_data=([Xu_valid, Xs_valid, Xc_valid], Y_tests),
			epochs=2, batch_size=32, verbose=1)
predics= model.predict([Xu_tests, Xs_tests, Xc_tests], verbose=1)
y_preds= np.argmax(predics, axis=1)
y_tests= np.argmax(Y_tests, axis=1)
matrix = confusion_matrix(y_tests, y_preds)
print('-'*55)
print(classification_report(y_tests, y_preds, target_names=classes))
print('Confusion Matrix:\n{}'.format(matrix))
model.save_weights('DeepClass.h5')
