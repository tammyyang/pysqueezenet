from keras.models import Graph, Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave
import theano
import cv2
import pdb
import os
import sys
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def squeeze():
	model = Graph()
	model.add_input(name='input',input_shape=(3,224,224))

	#conv 1
	model.add_node(Convolution2D(96, 3, 3, activation='relu', init='glorot_uniform',subsample=(2,2),border_mode='valid'),name='conv1', input='input')

	#maxpool 1
	model.add_node(MaxPooling2D((2,2)),name='maxpool1', input='conv1')

	#fire 1
	model.add_node(Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2_squeeze', input='maxpool1')
	model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2_expand1', input='fire2_squeeze')
	model.add_node(Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire2_expand2', input='fire2_squeeze')
	model.add_node(Activation("linear"),name='fire2', inputs=["fire2_expand1","fire2_expand2"], merge_mode="concat", concat_axis=1)

	#fire 2
	model.add_node(Convolution2D(16, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire3_squeeze', input='fire2')
	model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire3_expand1', input='fire3_squeeze')
	model.add_node(Convolution2D(64, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire3_expand2', input='fire3_squeeze')
	model.add_node(Activation("linear"),name='fire3', inputs=["fire3_expand1","fire3_expand2"], merge_mode="concat", concat_axis=1)

	#fire 3
	model.add_node(Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire4_squeeze', input='fire3')
	model.add_node(Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire4_expand1', input='fire4_squeeze')
	model.add_node(Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire4_expand2', input='fire4_squeeze')
	model.add_node(Activation("linear"),name='fire4', inputs=["fire4_expand1","fire4_expand2"], merge_mode="concat", concat_axis=1)

	#maxpool 4
	model.add_node(MaxPooling2D((2,2)),name='maxpool4', input='fire4')

	#fire 5
	model.add_node(Convolution2D(32, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5_squeeze', input='maxpool4')
	model.add_node(Convolution2D(128, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5_expand1', input='fire5_squeeze')
	model.add_node(Convolution2D(128, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire5_expand2', input='fire5_squeeze')
	model.add_node(Activation("linear"),name='fire5', inputs=["fire5_expand1","fire5_expand2"], merge_mode="concat", concat_axis=1)

	#fire 6
	model.add_node(Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6_squeeze', input='fire5')
	model.add_node(Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6_expand1', input='fire6_squeeze')
	model.add_node(Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire6_expand2', input='fire6_squeeze')
	model.add_node(Activation("linear"),name='fire6', inputs=["fire6_expand1","fire6_expand2"], merge_mode="concat", concat_axis=1)

	#fire 7
	model.add_node(Convolution2D(48, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7_squeeze', input='fire6')
	model.add_node(Convolution2D(192, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7_expand1', input='fire7_squeeze')
	model.add_node(Convolution2D(192, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire7_expand2', input='fire7_squeeze')
	model.add_node(Activation("linear"),name='fire7', inputs=["fire7_expand1","fire7_expand2"], merge_mode="concat", concat_axis=1)

	#fire 8
	model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8_squeeze', input='fire7')
	model.add_node(Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8_expand1', input='fire8_squeeze')
	model.add_node(Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire8_expand2', input='fire8_squeeze')
	model.add_node(Activation("linear"),name='fire8', inputs=["fire8_expand1","fire8_expand2"], merge_mode="concat", concat_axis=1)

	#maxpool 8
	model.add_node(MaxPooling2D((2,2)),name='maxpool8', input='fire8')

	#fire 9
	model.add_node(Convolution2D(64, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9_squeeze', input='maxpool8')
	model.add_node(Convolution2D(256, 1, 1, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9_expand1', input='fire9_squeeze')
	model.add_node(Convolution2D(256, 3, 3, activation='relu', init='glorot_uniform',border_mode='same'),name='fire9_expand2', input='fire9_squeeze')
	model.add_node(Activation("linear"),name='fire9', inputs=["fire9_expand1","fire9_expand2"], merge_mode="concat", concat_axis=1)
	model.add_node(Dropout(0.5),input='fire9',name='fire9_dropout')

	#conv 10
	model.add_node(Convolution2D(10, 1, 1, init='glorot_uniform',border_mode='valid'),name='conv10', input='fire9_dropout')
	#avgpool 1
	model.add_node(AveragePooling2D((13,13)),name='avgpool10', input='conv10')

	model.add_node(Flatten(),name='flatten',input='avgpool10')

	model.add_node(Activation("softmax"),input='flatten',name='softmax')

	model.add_output(name='output',input='softmax')

	return model

if __name__ == "__main__":

	model = squeeze()

	train_path = '/data/train/'
	val_path = '/data/val/'

	valid_ids = ['n07730033', 'n07734744', 'n07742313', 'n07745940', 
	'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 
	'n07754684']

	y_train = []
	y_test = []

	X_train = []
	X_test = []

	for idx, wnet_id in enumerate(valid_ids):

		# get train images
		train_path_curr = train_path + wnet_id
		for counter, filename in enumerate(os.listdir(train_path_curr)):
			if filename.endswith(".JPEG"): 
				img = cv2.imread(train_path_curr + '/' + filename)
				img = cv2.resize(img,(224,224))
				img = img.transpose((2,0,1))
				if counter%5 != 0:
					X_train.append(img)
					y_train.append(idx)
				else:
					X_test.append(img)
					y_test.append(idx)
				
		# get test images
		'''
		val_path_curr = val_path + wnet_id
		for filename in os.listdir(val_path_curr):
			if filename.endswith(".JPEG"): 
				img = cv2.imread(val_path_curr + '/' + filename)
				X_test.append(img)
				y_test.append(idx)
		'''
	X_train = np.asarray(X_train)
	X_test = np.asarray(X_test)
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)
	
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model.compile(optimizer=sgd, loss={'output':'categorical_crossentropy'})
	#model.compile(optimizer=sgd, loss='categorical_crossentropy')

	print(model.summary())

	# the data, shuffled and split between train and test sets
	#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	batch_size = 32
	nb_classes = 10
	nb_epoch = 200
	#data_augmentation = True

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	
	print('Not using data augmentation.')
	
	callback1 = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

	model.fit({'input':X_train,'output':Y_train}, batch_size=batch_size,nb_epoch=nb_epoch, validation_data={'input':X_test,'output':Y_test}, shuffle=True,callbacks=[callback1])
	#model.fit(X_train,Y_train, batch_size=batch_size,nb_epoch=nb_epoch, validation_data=(X_test,Y_test), shuffle=True,callbacks=[callback1])
