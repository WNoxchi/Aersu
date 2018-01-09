# Wayne Nixalo - 2018-Jan-08 19:02 - 2018-Jan-08 23:03
# Run Fast.ai classifiers on input data stream and display

# screen grab utility
from utils.getscreen import getScreen

# FastAI imports
from fastai_osx.model import resnet34
from fastai_osx.conv_learner import *


# Load dummy training set to init dataloader
dummy_fpath = 'data/train/000000-000412/000000.jpg'
train_dat = cv2.imread(dummy_fpath)
train_dat = cv2.cvtColor(train_dat, cv2.COLOR_BGR2RGB)
train_dat = np.array([train_dat]), np.array([1])
valid_dat = train_dat
classes   = [0,1]

# function to update dataloader with screengrab
def load_test_image(image=None):
    test_dat = np.array([image]) if type(image) == np.ndarray else None
    return ImageClassifierData.from_arrays(PATH, train_dat, valid_dat, bs=1,
                                           tfms=tfms, classes=classes, test=test_dat)

# Initialize FastAI Learner & load weights
PATH = 'data/'
sz = 400
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.2)
data = load_test_image()
learner = ConvLearner.pretrained(resnet34, data)
learner.load('RN34_400_WD_λ0-529_00')

# Darwin Retina system:
bbox = (8,160,682,544)
h = (bbox[3] - bbox[1]) * 2 # 768
w = (bbox[2] - bbox[0]) * 2 # 1348
tfx = 400 / w
tfy = 400 / h

# in_img = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
# in_img = cv2.cvtColor(in_img, cv2.COLOR_BGRA2RGB)

# video analysis loop
while True:
    # get & resize screengrab
    in_img = cv2.resize(np.asarray(getScreen(bbox=bbox)), None, fx=tfx, fy=tfy)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGRA2RGB)

    # load image into learner
    learner.set_data(load_test_image(in_img))

    # run image through classifier
    log_preds, _ = learner.TTA(is_test=True)

    # process prediction
    prediction = np.mean(np.exp(log_preds), 0)[0]

    # format prediction
    # prediction =

    # overlay prediction on copy of image. OpenCV uses BGR so convert for display
    out_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
    # black outline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    # white inline text:
    cv2.putText(out_img, f'{prediction}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # display image with prediction:
    cv2.imshow('GLOC Detector', out_img)

    # quit signal -- 'return':13, 'esc':27, 'q':ord('q')
    k = cv2.waitKey(1) & 0xFF
    if k in [13, 27, ord('q')]:
        break

# close OpenCV display window
cv2.destroyAllWindows()
