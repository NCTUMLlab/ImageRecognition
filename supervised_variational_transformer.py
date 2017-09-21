from __future__ import print_function

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Dense, merge, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Input, Activation, Reshape, Flatten
from keras.callbacks import LambdaCallback
from keras.objectives import mean_squared_error, categorical_crossentropy
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.utils import np_utils
from transform import spatial_transformer

#load shifted data
X_train_shift = np.load('data/X_train_shift_noise.npy')
X_test_shift  = np.load('data/X_test_shift_noise.npy')

#load mnist data
(X_train,y_train),(X_test,y_test)=mnist.load_data()

IMG_DIM = 100
if IMG_DIM == 100:
	#load cluttered data
	mnist_cluttered = np.load('data/mnist_sequence_sample_8distortions9x9.npz')
	X_train_cluttered, y_train_cluttered = mnist_cluttered['X_train'], mnist_cluttered['y_train']
	X_valid_cluttered, y_valid_cluttered = mnist_cluttered['X_valid'], mnist_cluttered['y_valid']
	X_test_cluttered, y_test_cluttered   = mnist_cluttered['X_test'], mnist_cluttered['y_test']
	y_train_cluttered=[int(x) for x in y_train_cluttered]
	y_valid_cluttered=[int(x) for x in y_valid_cluttered]
	y_test_cluttered=[int(x) for x in y_test_cluttered]
elif IMG_DIM == 60:
	#60x60
	mnist_cluttered = np.load('data/mnist_cluttered_60x60_6distortions.npz')
	X_train_cluttered, y_train_cluttered = mnist_cluttered['x_train'], np.argmax(mnist_cluttered['y_train'], axis=-1)
	X_valid_cluttered, y_valid_cluttered = mnist_cluttered['x_valid'], np.argmax(mnist_cluttered['y_valid'], axis=-1)
	X_test_cluttered, y_test_cluttered = mnist_cluttered['x_test'], np.argmax(mnist_cluttered['y_test'], axis=-1)

#load glimpse
X_train_glimpse = np.load('reconstruction/mnist_clutter_train_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM))
X_valid_glimpse = np.load('reconstruction/mnist_clutter_valid_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM))
X_test_glimpse  = np.load('reconstruction/mnist_clutter_test_glimpse_%dx%d.npy'%(IMG_DIM,IMG_DIM))

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
lw           = (1,1) #(classification,reconstruction)
nb_data      = X_train_shift.shape[0]
nb_valid     = 10000
batch_size   = 100
nb_batch     = int(nb_data/batch_size)
input_dim    = IMG_DIM*IMG_DIM
recon_dim    = 28*28
class_dim    = 10
att_h_1      = 200
att_h_2      = 200
att_dim      = 6
glimpse_size = (30,30)
class_h_1    = 200
class_h_2    = 200
epsilon_std  = 1.0
decode_h_1   = 200
decode_h_2   = 200
nb_sub_epoch = 5
nb_epoch     = 20


def sampling_normal(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(batch_size, att_dim), mean=0.,
	                                  stddev=epsilon_std)
	return z_mean + K.exp(z_log_var / 2) * epsilon


def transform(args):
	inputs,theta=args
	inputs = tf.reshape(inputs,[batch_size,IMG_DIM,IMG_DIM,1])
	theta  = tf.reshape(theta,[batch_size,6])
	img    = spatial_transformer(inputs,theta,glimpse_size)
	img    = tf.reshape(img,[batch_size,glimpse_size[0]*glimpse_size[1]])
	return img

b= np.zeros((2,3),dtype='float32')
b[0,0]=1
b[1,1]=1
W=np.zeros((att_h_2,6),dtype='float32')
weights = [W,b.flatten()]

#build model
x       = Input(batch_shape=(batch_size,input_dim))
y       = Input(batch_shape=(batch_size,class_dim))

with tf.device('/gpu:0'):
	conv1  = Reshape((IMG_DIM,IMG_DIM,1))(x)
	conv1  = Convolution2D(20,3,3)(conv1)
	conv1  = MaxPooling2D(pool_size=(2,2))(conv1)
	conv2  = Convolution2D(20,3,3)(conv1)
	conv2  = MaxPooling2D(pool_size=(2,2))(conv2)
	conv2  = Flatten()(conv2)

	att_h1 = Dense(att_h_1)(x)
	att_h1 = Activation('sigmoid')(att_h1)
	att_h2 = Dense(att_h_2)(att_h1)
	att_h2 = Activation('sigmoid')(att_h2)
	z_mean = Dense(att_dim,weights=weights,activation='tanh')(att_h2)
	#z_std  = Dense(att_dim)(att_h2)
	#z_sam1 = Lambda(sampling_normal)([z_mean,z_std])
	#z_sam2 = Lambda(sampling_normal)([z_mean,z_std])
	#z_sam3 = Lambda(sampling_normal)([z_mean,z_std])
	#z_nor  = merge([z_sam1,z_sam2,z_sam3],mode='ave')

with tf.device('/gpu:0'):
	img        = Lambda(transform)([x,z_mean])
	decode_h1  = Dense(decode_h_1)(img)
	decode_h1  = Activation('relu')(decode_h1)
	#decode_lb  = merge([decode_h1,y],mode='concat',concat_axis=1)
	decode_h2  = Dense(decode_h_2)(decode_h1)
	decode_h2  = Activation('relu')(decode_h2)
	pred_recon = Dense(recon_dim)(decode_h2)
	pred_recon = Activation('sigmoid',name='pred_recon')(pred_recon)

	conv1c     = Reshape((glimpse_size[0],glimpse_size[1],1))(img)
	conv1c     = Convolution2D(30,3,3)(conv1c)
	conv1c     = MaxPooling2D((2,2))(conv1c)
	conv2c     = Convolution2D(30,3,3)(conv1c)
	conv2c     = MaxPooling2D((2,2))(conv2c)
	conv2c     = Flatten()(conv2c)
	class_h1   = Dense(class_h_1)(conv2c)
	class_h1   = Activation('relu')(class_h1)
	class_h2   = Dense(class_h_2)(class_h1)
	class_h2   = Activation('relu')(class_h2)
	pred_y     = Dense(class_dim,activation='softmax',name='pred_y')(class_h2)
prior_model = Model([x,y],pred_recon)
class_model = Model([x,y],[pred_y,pred_recon])

class_model.compile(optimizer='adam',loss={'pred_y':'categorical_crossentropy','pred_recon':'mse'},metrics=['accuracy'],loss_weights={'pred_y':lw[0],'pred_recon':lw[1]})
prior_model.compile(optimizer='adam',loss='mse')



#Training
nb_valid_batch = int(nb_valid/batch_size)
best_acc       = 0.

#Training
for epoch in xrange(nb_epoch):
	print('='*120)
	print('Epoch: %d/%d'% (epoch+1,nb_epoch))
	print('*'*80)
	prior_model.fit([X_train_shift,Y_train],X_train,nb_epoch=nb_sub_epoch,batch_size=batch_size)		
	print('Sub-epoch is finished.')
	print('*'*80+'\n')
    	
	avg_cost  = 0.
	acc_train = 0.
	acc_valid = 0.
	hist=class_model.fit([X_train_shift,Y_train],[Y_train,X_train],shuffle=True,epochs=1,batch_size=batch_size,\
				validation_data=([X_valid_shift,Y_valid],[Y_valid,X_valid]))
	acc_valid = hist.history['val_pred_y_acc']
	acc_train = hist.history['pred_y_acc']
	if (acc_valid > best_acc) and (epoch >= 3) and (acc_train > acc_valid):
		best_acc=acc_valid
		class_model.save_weights('models/supervised_transform_shift_model.h5')
		print('Model is saved')
	print('\n')
print('='*120)
print("Training is finished")


#Test
class_model.load_weights('models/supervised_transform_shift_model.h5')
score = class_model.evaluate([X_test_shift,Y_test],[Y_test,X_test],batch_size = batch_size)
print('Test accuracy: %.4f'%(score[3]))

glimpe_model = Model(x,img)
img_att = glimpe_model.predict(X_test_shift,batch_size=100)
np.save('glimpse_shift_test_100.npy',img_att)
#att = Model(x,z_nor)
#att_mat = att.predict(X_test_shift,batch_size=batch_size)
#np.save('attention_mat/supervised_attention_mat_nokl.npy',att_mat)
#print('attention is saved')
#np.save('reconstruction/supervised_att_recon_test_nokl_2_noise.npy',test_recon)
#print('reconstruction is saved')
