#!/usr/bin/python3
import cgitb
import cgi
import base64
cgitb.enable(display=1)    
print("Content-Type: text/html;charset=utf-8\r\n")
form=cgi.FieldStorage()
fieldName='data'
if fieldName not in form: exit(0) 

postImage=form['data']
  
pngData=base64.b64decode(postImage.value.replace("data:image/png;base64,",''))
#png=postImage.file.read()
#print(png)

from PIL import Image
import io
pilImg=Image.open(io.BytesIO(pngData))
pilImg=pilImg.resize((28,28),Image.LANCZOS).convert(mode='F')
#print(pilImg)
import numpy as np
imgInput=np.absolute(np.asarray(pilImg).flatten()-255)/255
#print(imgInput) #dubuging with this will break the loading of dancing numbers
#################all glory to TF#####################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
session=tf.Session()
saver = tf.train.import_meta_graph('./myModel995.meta')
saver.restore(session,'./myModel995')
graph=tf.get_default_graph()
graphInput=graph.get_tensor_by_name('x:0')
graphOutput=graph.get_tensor_by_name('yCorrect:0')
evaluateInput=graph.get_tensor_by_name("predictResult:0")
feedDictTest = {graphInput:imgInput.reshape(1,784)}
result=np.zeros(shape=1,dtype=np.int)
result=session.run(evaluateInput,feed_dict=feedDictTest)
#print("Result:")
print(result[0])
#print("---------------------")

