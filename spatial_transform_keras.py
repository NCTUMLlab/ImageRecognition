from __future__ import print_function

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Dense, merge, Lambda, Flatten
from keras.layers import Activation, Input, Reshape, Convolution2D, MaxPooling2D
from keras.models import Model
from keras.objectives import mean_squared_error, categorical_crossentropy
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from transform import spatial_transformer



#load shifted data
X_train_shift   = np.load('data/X_train_shift_noise.npy')
X_test_shift    = np.load('data/X_test_shift_noise.npy')
#mnist_cluttered = np.load('data/mnist_cluttered_60x60_6distortions.npz')
mnist_cluttered = np.load('data/mnist_sequence_sample_8distortions9x9.npz')
IMG_DIM = 100

#load cluttered data
if IMG_DIM == 100:
	X_train_cluttered, y_train_cluttered = mnist_cluttered['X_train'], mnist_cluttered['y_train']
	X_valid_cluttered, y_valid_cluttered = mnist_cluttered['X_valid'], mnist_cluttered['y_valid']
	X_test_cluttered, y_test_cluttered   = mnist_cluttered['X_test'], mnist_cluttered['y_test']
	y_train_cluttered=[int(x) for x in y_train_cluttered]
	y_valid_cluttered=[int(x) for x in y_valid_cluttered]
	y_test_cluttered=[int(x) for x in y_test_cluttered]
elif IMG_DIM == 60:
	#60x60
	X_train_cluttered, y_train_cluttered = mnist_cluttered['x_train'], np.argmax(mnist_cluttered['y_train'], axis=-1)
	X_valid_cluttered, y_valid_cluttered = mnist_cluttered['x_valid'], np.argmax(mnist_cluttered['y_valid'], axis=-1)
	X_test_cluttered, y_test_cluttered = mnist_cluttered['x_test'], np.argmax(mnist_cluttered['y_test'], axis=-1)

#load mnist data
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#mnist data preprocess
X_train = X_train.astype('float32') / 255.
X_test  = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test  = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test, 10)
X_valid = X_train[50000:]
X_train = X_train[:50000]
Y_valid = Y_train[50000:]
Y_train = Y_train[:50000]

#shifted data preprocess
X_train_shift = X_train_shift.astype('float32')/255.
X_test_shift  = X_test_shift.astype('float32')/255.
X_train_shift = X_train_shift.reshape((len(X_train_shift), np.prod(X_train_shift.shape[1:])))
X_test_shift  = X_test_shift.reshape((len(X_test_shift), np.prod(X_test_shift.shape[1:])))
X_valid_shift = X_train_shift[50000:]
X_train_shift = X_train_shift[:50000]

#cluttered data preprocess
X_train_cluttered = X_train_cluttered.reshape((X_train_cluttered.shape[0], IMG_DIM*IMG_DIM))
X_valid_cluttered = X_valid_cluttered.reshape((X_valid_cluttered.shape[0], IMG_DIM*IMG_DIM))
X_test_cluttered  = X_test_cluttered.reshape((X_test_cluttered.shape[0], IMG_DIM*IMG_DIM))
Y_train_cluttered = np_utils.to_categorical(y_train_cluttered, 10)
Y_valid_cluttered = np_utils.to_categorical(y_valid_cluttered, 10)
Y_test_cluttered  = np_utils.to_categorical(y_test_cluttered, 10)

#parameters
input_dim  = IMG_DIM*IMG_DIM
output_dim = 10
batch_size = 100
local_h_1  = 200
local_h_2  = 200
n_theta    = 6
class_h_1  = 200
class_h_2  = 200
nb_epoch   = 50

#initial weight
b = np.zeros((2,3),dtype='float32')
b[0,0]=1.
b[1,1]=1.
W = np.zeros((local_h_2,6),dtype='float32')
weights = [W,b.flatten()]

def transform(args):
	inputs,theta=args
	inputs = tf.reshape(inputs,[batch_size,IMG_DIM,IMG_DIM,1])
	theta  = tf.reshape(theta,[batch_size,6])
	img = spatial_transformer(inputs,theta,(30,30))
	img = tf.reshape(img,[batch_size,30*30])
	return img

#build model
x  = Input(batch_shape=(batch_size,input_dim))

conv1 = Reshape((IMG_DIM,IMG_DIM,1))(x)
conv1 = Convolution2D(20,5,5)(conv1)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv2 = Convolution2D(20,5,5)(conv1)
conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv2 = Flatten()(conv2)

local_h1 = Dense(local_h_1,activation='sigmoid')(conv2)
local_h2 = Dense(local_h_2,activation='sigmoid')(local_h1)
theta    = Dense(n_theta,weights=weights,activation='tanh')(local_h2)
img      = Lambda(transform)([x,theta])

conv1c   = Reshape((30,30,1))(img)
conv1c   = Convolution2D(20,3,3)(conv1c)
conv1c   = MaxPooling2D((2,2))(conv1c)
conv2c   = Convolution2D(20,3,3)(conv1c)
conv2c   = MaxPooling2D((2,2))(conv2c)
conv2c   = Flatten()(conv2c)
class_h1 = Dense(class_h_1,activation='relu')(conv2c)
class_h2 = Dense(class_h_2,activation='relu')(class_h1)
logit    = Dense(output_dim,activation='softmax')(class_h2)

st_model = Model(x,logit)
st_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
st_model.fit(X_train_shift,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_valid_shift,Y_valid))
score=st_model.evaluate(X_test_cluttered,Y_test_cluttered,batch_size=batch_size)

print("\nTest accuracy: %.4f"%(score[1]))
generator = Model(x,img)
pic_train = generator.predict(X_train_shift,batch_size=batch_size)
pic_valid = generator.predict(X_valid_shift,batch_size=batch_size)
pic_test  = generator.predict(X_test_shift,batch_size=batch_size)
np.save('reconstruction/mnist_shift_train_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM),pic_train)
np.save('reconstruction/mnist_shift_valid_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM),pic_valid)
np.save('reconstruction/mnist_shift_test_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM),pic_test)
