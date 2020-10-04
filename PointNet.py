import h5py
import keras
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

with h5py.File('Y.hdf5'    , 'r') as yh: Y = yh['default'][()]
with h5py.File('Coord.hdf5', 'r') as ch: Coord = ch['default'][()]

X_train, X_tests, Y_train, Y_tests = train_test_split(Coord, Y, test_size=0.20)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25)

print(X_train.shape, X_valid.shape, X_tests.shape)
print(Y_train.shape, Y_valid.shape, Y_tests.shape)

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
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['sparse_categorical_accuracy'])
model.fit(X_train, Y_train,
    epochs=100,
    validation_data=(X_valid, Y_valid),
    batch_size=32)

classes = ['Alpha', 'Not_Alpha']
evaluation = model.evaluate(X_tests, Y_tests)
print('Test Set: Accuracy {} Loss {}'\
.format(round(evaluation[1], 4), round(evaluation[0], 4)))
Y_pred = model.predict(X_tests)
y_pred = np.argmax(Y_pred, axis=1)
matrix = confusion_matrix(Y_tests, y_pred)
df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
print(classification_report(Y_tests, y_pred, target_names=classes))
model.save_weights('weights.h5')
