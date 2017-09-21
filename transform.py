import tensorflow as tf
import numpy as np

def DNN(x,input_dim,output_dim,h_dim,prob,scope='DNN'):
######################################################
# Define the DNN model to estimate the transform parameters
# input:
# 	x: n x D Tensor,n is the batch size, D is the input
#	   dimension
#	input_dim: scalar, input dimension is 1600 in 
#		   the model
#	output_dim: scalar, output dimension is 6 in
#		    the model
#	h_dim: scalar, the dimension in every hidden layer
#	prob: scalar, the keep probability of dropout
#	scope: string, tensorflow variable scope
# output: 
#	out: n x output_dim Tensor, the estimation for the 
#	     transform parameters
#
######################################################
	with tf.variable_scope(scope):
		# Create learnable variable for DNN
		w1=tf.Variable(tf.zeros([input_dim,h_dim]))
		b1=tf.Variable(tf.random_normal([h_dim],mean=0,stddev=0.01))

		w2=tf.Variable(tf.zeros([h_dim,h_dim]))
		b2=tf.Variable(tf.random_normal([h_dim],mean=0,stddev=0.01))
		initial = np.array([[1., 0, 0], [0, 1., 0]])
		initial = initial.astype('float32')
		initial = initial.flatten()

		w_out=tf.Variable(tf.zeros([h_dim,output_dim]))
		b_out=tf.Variable(initial_value=initial)
		
		# Define the DNN structure
		h1=tf.add(tf.matmul(x,w1),b1)
		h1=tf.nn.sigmoid(h1)
		h1=tf.nn.dropout(h1,prob)

		h2=tf.add(tf.matmul(h1,w2),b2)
		h2=tf.nn.tanh(h2)
		h2=tf.nn.dropout(h2,prob)

		out=tf.add(tf.matmul(h2,w_out),b_out)
		out=tf.nn.tanh(out)
	
	return out

def conv2d(x,W,b,stride=1):
####################################################
# Define the convolution layer
# input:
# 	x: n x h x w Tensor, the batch of images,
#	   n is the batch size, h is the height of 
#	   the images, w is the width of images
#	W: Variable, weight variable
#	b: Variable, bias variable
#	stride: scalar, the movement of the filter
# output:
# 	out: a Tensor which has been computed by the
#	     convolution layer
#
###################################################
	x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
	x=tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

def maxpool2d(x,k=2):
####################################################
# Define the max pooling layer
# input:
# 	x: n x h x w Tensor, the batch of images,
#	   n is the batch size, h is the height of 
#	   the images, w is the width of images
#	stride: scalar, the movement of the filter
# output:
# 	out: a Tensor which has been computed by the
#	     max pooling layer
#
###################################################
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def CNN(x,output_dim,input_shape,prob,scope='CNN'):
######################################################
# Define the CNN model to recognize the image
# input:
# 	x: n x h x w Tensor, the batch of images,
#	   n is the batch size, h is the height of 
#	   the images, w is the width of images
#	output_dim: scalar, output dimension is 10 in
#		    the model
#	prob: scalar, the keep probability of dropout
#	scope: string, tensorflow variable scope
# output: 
#	out: n x output_dim Tensor, n is the batch size,
#        it is the result of the recognition
#
######################################################
	height=input_shape[0]
	width=input_shape[1]
	with tf.variable_scope(scope):
        # Create the learnable variable
		wc1=tf.Variable(tf.random_normal([3,3,1,16]))
		bc1=tf.Variable(tf.random_normal([16]))

		wc2=tf.Variable(tf.random_normal([3,3,16,16]))
		bc2=tf.Variable(tf.random_normal([16]))
		
		wf1=tf.Variable(tf.random_normal([(height/4)*(width/4)*16,1024]))
		bf1=tf.Variable(tf.random_normal([1024]))

		wf2=tf.Variable(tf.random_normal([1024,output_dim]))
		bf2=tf.Variable(tf.random_normal([output_dim]))
        
        
        # Define the structure of the CNN model 
		conv1=conv2d(x,wc1,bc1)
		conv1=maxpool2d(conv1,k=2)
		conv2=conv2d(conv1,wc2,bc2)
		conv2=maxpool2d(conv2,k=2)
        # Reshape the input from a matrix to a vector
		fc1=tf.reshape(conv2,[-1,(height/4)*(width/4)*16])
        # Define the fully-connect layer
		fc1=tf.add(tf.matmul(fc1,wf1),bf1)
		fc1=tf.nn.relu(fc1)
		fc1=tf.nn.dropout(fc1,prob)

		fc2=tf.add(tf.matmul(fc1,wf2),bf2)
	return fc2

def _meshgrid(height,width):
###############################################
# Create a grid 
# input:
#	height: scalar, the height of grid
#	width: scalar, the width of grid
# output:
#	out: the grid, a list [0,0,1] to [1,1,1]
#
################################################
	xv = tf.matmul(tf.ones(shape=tf.stack([height, 1])),tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
	yv = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),tf.ones(shape=tf.stack([1, width])))

	xv_flat=tf.reshape(xv,(1,-1))
	yv_flat=tf.reshape(yv,(1,-1))

	ones = tf.ones_like(xv_flat)

	grid = tf.concat(axis=0, values=[xv_flat, yv_flat, ones])
	return grid

def _repeat(x, n_repeats):
##########################################################
# Repeat input certain times
# input:
#	x: the input image
#	n_repeat: scalar, the repeat times
# output:
#	out:the vector which contains x with n_repeat times 
#
###########################################################
      with tf.variable_scope('_repeat'):
                rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
		rep = tf.cast(rep, 'int32')
		x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
		return tf.reshape(x, [-1])

def _interpolate(im,x,y,out_size):
##########################################
# Generate the glimpse
# input:
#	im: the input images
#	x: the x coordinate of grid
#	y: the y coordinate of grid
#	out_size=the glimpse size
# output:
#	out: the glimpse, a part of im
#
##########################################
	with tf.variable_scope('_interpolate'):
		n_batch=tf.shape(im)[0]
		height=tf.shape(im)[1]
		width=tf.shape(im)[2]
		channels=tf.shape(im)[3]
		out_height=out_size[0]
		out_width=out_size[1]
		height_f=tf.cast(height,'float32')
		width_f=tf.cast(width,'float32')
		x=tf.cast(x,'float32')
		y=tf.cast(y,'float32')
		max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
		max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
		zero = tf.zeros([], dtype='int32')

		# transform [-1,1] to [0,width or height]
		x=(x+1.0)*width_f/2.0
		y=(y+1.0)*height_f/2.0

		# Sample
		x0=tf.cast(tf.floor(x),'int32')
		x1=x0+1
		y0=tf.cast(tf.floor(y),'int32')
		y1=y0+1
		
		x0=tf.clip_by_value(x0,zero,max_x)
		x1=tf.clip_by_value(x1,zero,max_x)
		y0=tf.clip_by_value(y0,zero,max_y)
		y1=tf.clip_by_value(y1,zero,max_y)

		dim2=width
		dim1=width*height
		base=_repeat(tf.range(n_batch)*dim1,out_height*out_width)
		base_y0=base+y0*dim2
		base_y1=base+y1*dim2
		idx_a=base_y0+x0
		idx_b=base_y1+x0
		idx_c=base_y0+x1
		idx_d=base_y1+x1
		# use indices to lookup pixels in the flat image
		im_flat = tf.reshape(im, tf.stack([-1, channels]))
		im_flat = tf.cast(im_flat, 'float32')
                Ia = tf.gather(im_flat, idx_a)
		Ib = tf.gather(im_flat, idx_b)
		Ic = tf.gather(im_flat, idx_c)
		Id = tf.gather(im_flat, idx_d)

		# and finally calculate interpolated values
		x0_f = tf.cast(x0, 'float32')
		x1_f = tf.cast(x1, 'float32')
		y0_f = tf.cast(y0, 'float32')
		y1_f = tf.cast(y1, 'float32')
		wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
		wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
		wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
		wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
		output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
		return output




def spatial_transformer(U,theta,output_size,scope='spatial'):
###############################################################
# Define a funciton to get the glimpse from the image base
# on the given transform parameters
# input:
#	U: the input images(Tensor type)
#	theta: n x 6, the transform parameters, n is the batch
#	       size.
#	output_size: the output_size of the spatial transformer
#	scope: string, tensorflow variable scope
# output:
#	out: the glimpse with the output_size
#
###############################################################
	with tf.variable_scope(scope):
		# Get constants from the inputs
		n_batch=tf.shape(U)[0]
		height=tf.shape(U)[1]
		width=tf.shape(U)[2]
		n_channel=tf.shape(U)[3]
		theta=tf.reshape(theta,(-1,2,3))
		theta=tf.cast(theta,'float32')
		height_out=output_size[0]
		width_out=output_size[1]

		# Create a grid to generate the glimpse
		grid=_meshgrid(height_out,width_out)
		grid=tf.expand_dims(grid,0)
		grid=tf.reshape(grid,[-1])
		grid=tf.tile(grid, tf.stack([n_batch]))
		grid=tf.reshape(grid,tf.stack([n_batch,3,-1]))

		# Transform Ax(x_t, y_t, 1)^T -> (x_s, y_s)
		T_g=tf.matmul(theta,grid)
		x_s=tf.slice(T_g,[0,0,0],[-1,1,-1])
		y_s=tf.slice(T_g,[0,1,0],[-1,1,-1])
		x_s_flat=tf.reshape(x_s,[-1])
		y_s_flat=tf.reshape(y_s,[-1])
		# Generate the glimpse
		input_trans=_interpolate(U,x_s_flat,y_s_flat,output_size)
		out=tf.reshape(input_trans,tf.stack([n_batch,height_out,width_out,n_channel]))
	return out
