#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/mnist/',one_hot=True)

session=tf.Session()
saver = tf.train.import_meta_graph('./myModel99.meta')
saver.restore(session,tf.train.latest_checkpoint('./'))
graph=tf.get_default_graph()
graphInput=graph.get_tensor_by_name('x:0')
graphOutput=graph.get_tensor_by_name('yCorrect:0')
print(graphInput)
print(graphOutput)

evaluateInput=graph.get_tensor_by_name("predictResult:0")
print(evaluateInput)

for i in range(10):
        testBatch,testCorrect = data.test.next_batch(1)
        feedDictTest = {graphInput:testBatch}
        print("Correct Result (0-9):")
        print(session.run(tf.argmax(testCorrect,axis=1))[0])
        #rint(feedDictTest)
        result=np.zeros(shape=1,dtype=np.int)
        result=session.run(evaluateInput,feed_dict=feedDictTest)
        print("Result:")
        print(result[0])
        print("---------------------")

