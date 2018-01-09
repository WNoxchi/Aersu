# Wayne Nixalo - 2018-Jan-08 19:02 - 2018-Jan-08 23:03
# Run Fast.ai classifiers on input data stream and display

# screen grab utility
from utils.getscreen import getScreen

# FastAI imports
from fastai_osx.model import resnet34
from fastai_osx.conv_learner import *

# Common & Keras-RetinaNet imports
from utils.common import *
import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
import tensorflow as tf

show_time = False

if show_time: t = time.time()
# Initialize Keras (TF) RetinaNet model
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # will this eat all GRAM?
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())
model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                custom_objects=custom_objects)

# Load dummy training set to init Fast.ai dataloader
train_dat = load_dummy(fpath='data/train/000000-000412/000000.jpg')
valid_dat = train_dat

# Initialize FastAI (PT) Learner & load weights
PATH = 'data/'
sz = 400
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.2)
data = load_test_image(PATH, train_dat=train_dat, valid_dat=valid_dat, tfms=tfms)
learner = ConvLearner.pretrained(resnet34, data)
learner.load('RN34_400_WD_λ0-529_00')

# Darwin Retina system:
bbox = (8,160,682,544)
h = (bbox[3] - bbox[1]) * 2 # 768
w = (bbox[2] - bbox[0]) * 2 # 1348
tfx = sz / w
tfy = sz / h

print(f'T(INITIALIZE): {time.time() - t:.2f}')

# video analysis loop
while True:
    # get & resize screengrab
    in_img = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGRA2RGB)

                ### 1st STAGE: LOCATOR

    if show_time: t = time.time()

    # detect & crop pilot
    bounding_box = detect(in_img, threshold=0., mode='auto', model=model)
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

    # overlay prediction on copy of image. OpenCV uses BGR so convert for display
    out_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)

    # draw bounding box:
    b = bounding_box
    cv2.rectangle(out_img, (b[0], b[1]), (b[2], b[3]), (255,0,0), 3)

    # black outline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    # white inline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # display image with prediction:
    cv2.imshow('GLOC Detector', out_img)

    if show_time: print(f'T(STG-2): {time.time()-t:.2f}')

    # quit signal -- 'return':13, 'esc':27, 'q':ord('q')
    k = cv2.waitKey(1) & 0xFF
    if k in [13, 27, ord('q')]:
        break

# close OpenCV display window
cv2.destroyAllWindows()
