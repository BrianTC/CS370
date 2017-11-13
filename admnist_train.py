#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/mnist/',one_hot=True)
#print("Size of:")
#print("Training data {}".format(len(data.train.labels)))
img_size = 28
img_size_flat = img_size ** 2
#print(img_size_flat)
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
def newWeights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def newBiases(length):
	return tf.Variable(tf.constant(0.05,shape=[length]))

def newConvolutionLayer(input,
			numInputChannels,
			filterSize,
			numFilters,
			usePooling=True):
	shape = [filterSize,filterSize,numInputChannels,numFilters]
	weights = newWeights(shape=shape)
	biases = newBiases(length=numFilters)
	layer = tf.nn.conv2d(input=input,
			filter=weights,
			strides=[1,1,1,1],
			padding='SAME')
	layer+=biases
	if usePooling:
		layer=tf.nn.max_pool(value=layer,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding='SAME')
	layer = tf.nn.relu(layer)
	return layer

def flattenLayer(layer):
	layerShape= layer.get_shape()
	numFeatures = layerShape[1:4].num_elements()
	layerFlat = tf.reshape(layer,[-1,numFeatures])
	return layerFlat

def newFCLayer(input,
		numInputs,
		numOutputs,
		useRELU=True):
	weights = newWeights(shape=[numInputs,numOutputs])
	biases = newBiases(length=numOutputs)
	layer=tf.matmul(input,weights)+biases
	if useRELU:
		layer=tf.nn.relu(layer)
	return layer

x = tf.placeholder(tf.float32, shape =[None,img_size_flat],name='x')
xImg = tf.reshape(x,[-1,img_size,img_size,num_channels])
yCorrect = tf.placeholder(tf.float32,shape=[None,num_classes],name='yCorrect')

yCorrectClass = tf.argmax(yCorrect,axis=1)

#layer1
filterSize_1=5
numFilters_1=16
layerConv1= newConvolutionLayer(input=xImg,
				numInputChannels=num_channels,
				filterSize=filterSize_1,
				numFilters=numFilters_1,
				usePooling=True)
#print(layerConv1)
#layer2
filterSize_2=5
numFilters_2=36
layerConv2= newConvolutionLayer(input=layerConv1,
				numInputChannels=numFilters_1,
				filterSize=filterSize_2,
				numFilters=numFilters_2,
				usePooling=True)
#print(layerConv2)

#flatten layer2 for fullyConnected Layer

layerFlat = flattenLayer(layerConv2)
#print(layerFlat)
#print(layerFlat.shape[1])
#FCL1
fcSize=128
layerFc1=newFCLayer(input=layerFlat,
			numInputs=1764,#layerFlat.shape[1],
			numOutputs=fcSize,
			useRELU=True)
#FC2 output layer
layerFc2=newFCLayer(input=layerFc1,
			numInputs=fcSize,
			numOutputs=num_classes,
			useRELU=True)
#format output
yPredicted=tf.nn.softmax(layerFc2)
yPredClass=tf.argmax(yPredicted,axis=1,name='predictResult')

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layerFc2,labels=yCorrect))
#grad dec aglo
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,epsilon=1e-22).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correctPrediction=tf.equal(yPredClass,yCorrectClass)
accuracy=tf.reduce_mean(tf.cast(correctPrediction,tf.float32))

#start session
saver=tf.train.Saver()
session=tf.Session()
session.run(tf.global_variables_initializer())

total_iterations = 0

train_batch_size=256
testPoint=0.7
def optimize(num_iterations):
	global total_iterations
	start_time = time.time()

	for i in range(total_iterations,total_iterations + num_iterations):
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		feed_dict_train = {x: x_batch,yCorrect: y_true_batch}
		session.run(optimizer, feed_dict=feed_dict_train)
		if i % 100 == 0:
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			if acc > testPoint:
				print("Acc > testPoint ("+str(testPoint)+")")
				#use a large sample size to test model
				testBatch,testCorrect = data.test.next_batch(2048)
				feedDictTest = {x:testBatch, yCorrect:testCorrect}
				testAcc=session.run(accuracy,feed_dict=feedDictTest)
				print("Test accuracy: " + str(testAcc*100))
				if testAcc > testPoint: 	
					print("Time to save")
					saver.save(session,'./myModel_inClass')
					break; 
			#print(acc)
			print(msg.format(i + 1, acc))
			
	total_iterations += num_iterations
    	# Ending time.
	end_time = time.time()
    	# Difference between start and end-times.
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
	

optimize(1)

optimize(100000)
for i in range(10):
	testBatch,testCorrect = data.test.next_batch(1)
	feedDictTest = {x:testBatch}
	print("Correct Result (0-9):")
	print(session.run(tf.argmax(testCorrect,axis=1))[0])
	#rint(feedDictTest)
	result=np.zeros(shape=1,dtype=np.int)
	result=session.run(yPredClass,feed_dict=feedDictTest)
	print("Result:")
	print(result[0])
	print("---------------------")
