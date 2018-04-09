# # Wayne Nixalo - 9-Jan-2018 22:38
# # what the fuck is wrong with tensorflow gpu
#
# # I'm attempting to see exactly which point triggers the
# # *** stack smashing detected *** Aborted (core dumped) error...
#
# screen grab utility
from utils.getscreen import getScreen



# Common & Keras-RetinaNet imports
from utils.common import *
import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
import tensorflow as tf

# FastAI imports
from fastai.model import resnet34
from fastai.conv_learner import *

# Initialize Keras (TF) RetinaNet model
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())
model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                custom_objects=custom_objects)
print(type(model))


# from utils.getscreen import getScreen

sz = 400

bbox = (60,60,1060,660)
h,w = (bbox[3]-bbox[1]),(bbox[2]-bbox[0])
tfx = sz/w
tfy = sz/h


image = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

# print(model.predict(np.array([image]))[0])

t = 0.
t1= 0.

for i in range(10):
    t1 = time.time()
    bounding_box = detect(image, threshold=0., mode='auto', model=model)
    print(bounding_box)
    t += time.time()-t1

print(f'Time (avg 10): {t/10:.2f}')


                        #### SUCCESSFUL LOAD BELOW

# import keras
# import tensorflow as tf
#
# import cv2
# import numpy as np
#
#
# def get_session():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth=True
#     return tf.Session(config=config)
#
# keras.backend.tensorflow_backend.set_session(get_session())
#
# # model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#
# from utils.getscreen import getScreen
#
# sz = 400
#
# bbox = (60,60,1060,660)
# h,w = (bbox[3]-bbox[1]),(bbox[2]-bbox[0])
# tfx = sz/w
# tfy = sz/h
#
#
#
# image = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
# image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
#
# # model.predict(np.array([image]))
#
# import keras.preprocessing.image
# from keras_retinanet.models.resnet import custom_objects
#
# model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5', custom_objects=custom_objects)
#
# print(type(model))
