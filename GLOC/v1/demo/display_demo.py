# WNixalo Edited: 2018-May-09 20:29
#

import sys
import os
sys.path.insert(1, os.path.join('../'))

# Wayne Nixalo - 2018-Jan-08 19:02 - 2018-Jan-08 23:03
# Run Fast.ai classifiers on input data stream and display

# Keras-RetinaNet imports
import keras
import keras.preprocessing.image
# from keras_retinanet.models.resnet import custom_objects
from keras_retinanet import models
import tensorflow as tf

# FastAI imports
from fastai.model import resnet34
from fastai.conv_learner import *

# Common Imports (GLOC)
from utils.getscreen import getScreen
# from utils.common import *
from demo_utils.common import *

print(f'Starting GLoC Detector V1. To quit: press "q" while the image window is '
      f'selected. Program will quit after the next update. (you may need to '
      f' press "q" a few times)\n')

show_time = True
fullsize_out = True

if show_time: t = time.time()
# Initialize Keras (TF) RetinaNet model
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

# import pdb; pdb.set_trace()

keras.backend.tensorflow_backend.set_session(get_session())
# model = keras.models.load_model('data/retinanet-model/resnet50_coco_best.h5',
#                                 custom_objects=custom_objects)

# 2018-May-09 20:56 fix to bug in fizyr custom keras layer: https://github.com/fizyr/keras-retinanet/issues/407#issuecomment-382312153
# example of fix in § 'Load RetinaNet model': https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
model_path = 'demo_data/retinanet-model/resnet50_coco_best.h5'
# model = models.load_model(model_path, backbone_name='resnet50', convert=True)
model = models.load_model(model_path, backbone_name='resnet50')

# Get path to first file in dataset to initialize fast.ai dataloader
first_subfolder = os.listdir('demo_data/train')
first_subfolder.sort()
if '.DS_Store' in first_subfolder: first_subfolder.remove('.DS_Store')
first_subfolder = first_subfolder[0]
first_image = os.listdir('demo_data/train/' + first_subfolder)
first_image.sort()
first_image = first_image[0]
dummy_path = 'demo_data/train/' + first_subfolder + '/' + first_image

# Load dummy training set to init Fast.ai dataloader
train_dat = load_dummy(fpath=dummy_path)
# train_dat = load_dummy(fpath=f'data/train/000000-000412/000000.jpg')
valid_dat = train_dat

# Initialize FastAI (PT) Learner & load weights
PATH = 'demo_data/'
sz = 400
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.2)
data = load_test_image(PATH, train_dat=train_dat, valid_dat=valid_dat, tfms=tfms)
learner = ConvLearner.pretrained(resnet34, data)
# learner.load('RN34_400_WD_λ0-529_00')
learner.load('dummy_resnet34_classifier')

# Darwin Retina system:
if sys.platform[:3] == 'dar':
    bbox = (8,160,682,544)
    s    = 2
# Linux/Windows Non-Retina:
else:
    bbox = (65,185,65+918,185+522)
    s    = 1
h = (bbox[3] - bbox[1]) * s # 768
w = (bbox[2] - bbox[0]) * s # 1348
tfx = sz / w
tfy = sz / h

if show_time: print(f'T(INITIALIZE): {time.time() - t:.2f}')

# quick bugfix: count consecutive non-detections
non_detect = 0

# video analysis loop
while True:
    # get & resize screengrab
    if fullsize_out:
        img = np.asarray(getScreen(bbox=bbox))
        in_img = cv2.resize(img, None, fx=tfx, fy=tfy)
    else:
        in_img = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_RGBA2RGB)

                ### 1st STAGE: LOCATOR

    if show_time: t = time.time()

    # detect & crop pilot
    bounding_box = detect(in_img, threshold=0., mode='auto', model=model)

    if type(bounding_box) == int:
        if show_time:
            print(f'No detection')
        # bugfix: quit automatically after 4 consecutive non-detections
        non_detect += 1
        if non_detect >= 4:
            print(f'4 Consecutive Non-Detections: Quitting.')
            break
        continue
    else:
        non_detect = 0

    crop_img = crop(in_img, bounding_box)

    # overlay crop on center of black bg (background)
    bg = in_img.copy()
    bg[:] = 0
    xof = (bg.shape[1] - crop_img.shape[1])//2
    yof = (bg.shape[0] - crop_img.shape[0])//2
    overlay = overlay_image(bg, crop_img, xof, yof)

    if show_time: print(f'T(STG-1): {time.time()-t:.2f}')

                ### 2nd STAGE: DETECTOR

    if show_time: t = time.time()

    # load image into learner
    learner.set_data(load_test_image(PATH, overlay, train_dat=train_dat,
                                     valid_dat=valid_dat, tfms=tfms)
                    )
    # run image through classifier
    log_preds, _ = learner.TTA(is_test=True)

    # process prediction
    prediction = np.mean(np.exp(log_preds), 0)[0] # = [[pred_0, pred_1]][0]

    # format prediction
    # prediction =

    # overlay prediction on copy of image
    out_img = img if fullsize_out else in_img

    # draw bounding box:
    b = bounding_box
    if fullsize_out:
        b = [int(c/tfx) if i % 2 == 0 else int(c/tfy) for i,c in enumerate(b)]
    cv2.rectangle(out_img, (b[0], b[1]), (b[2], b[3]), (255,0,0), 3)

    # black outline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    # white inline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    # display image with prediction:
    cv2.imshow('GLOC Detector', out_img)

    if show_time: print(f'T(STG-2): {time.time()-t:.2f}')

    # quit signal -- 'return':13, 'esc':27, 'q':ord('q')
    k = cv2.waitKey(1) & 0xFF
    if k in [13, 27, ord('q')]:
        break

# close OpenCV display window
cv2.destroyAllWindows()
