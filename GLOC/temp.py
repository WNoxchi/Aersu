%matplotlib inline
%reload_ext autoreload
%autoreload 2


from fastai_osx.model import resnet34
from fastai_osx.conv_learner import *


PATH = 'data/'

def get_image_ndarray(path='data/train/'):
    """Returns a random image as an ndarray in an ndarray from the GLOC Dataset"""
    # get random image
    if '.DS_Store' in os.listdir(path):
        os.remove(path + '.DS_Store')
    folders = os.listdir(path)
    folder  = np.random.choice(folders)
    fname   = np.random.choice(os.listdir(path+folder))
    fpath   = path+folder+'/'+fname; fpath

    # load image as ndarray
    image = cv2.imread(fpath) # dtype = 'uint8'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB conversion

    # reshape image to PyTorch Tensor Order: (Channel,H,W)
    # image = np.rollaxis(image, 2, 0)

    # return image as ndarray of ndarrays, and image filepath
    return np.array([image]), fpath


# get random test image and dummy train/val image & label
img_ndarray, fpath = get_image_ndarray()

train_dat = img_ndarray, np.array([1])
val_dat = train_dat

test_dat, fpath = get_image_ndarray()

classes = [0,1]


# initialize dataloader and learner
sz = 400
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.2)
data = ImageClassifierData.from_arrays(PATH, train_dat, val_dat, bs=1,
                                       tfms=tfms, classes=classes, test=test_dat)
learner = ConvLearner.pretrained(resnet34, data)
learner.load('RN34_400_WD_λ0-529_00')


# run neural net on image
logpred,_ = learner.TTA(is_test=True)
pred = np.mean(np.exp(logpred), 0)


# get actual label from CSV
label_df = pd.read_csv(PATH + 'labels.csv')
folder   = fpath.split('/')[-2]
fname    = fpath.split('/')[-1]
answer   = labels_df.loc[labels_df['id']==folder+'/'+fname.split('.')[0]]['gloc'].values[0]


# display results
print(fname)
print(pred)
print(answer)
plt.imshow(img_ndarray[0])





























from fastai_osx.model import resnet34
from fastai_osx.conv_learner import * # imports learner -> imports dataset & imports


# PyTorch Tensor shape: (N, Channels, Rows, Cols) # or (,,X,Y)?
train_dat = (np.array([]), np.array([]))

sz = 400
tfms = tfms_from_model(resnet34, sz)
# ImageClassifierData.from_arrays(path, trn, val, bs=64, tfms=(None, None),
#                                   classes=None, num_workers=4, test=None)
data = ImageClassifierData.from_arrays(PATH, train_dat, val_dat, bs=1,
                                       tfms=tfms, classes=classes, test=test_dat)






























import fastai_osx.dataset



data = ImageClassifierData.from_csv(PATH, 'train', labels_csv, bs=bs, tfms=tfms,
                                    val_idxs=val_idxs, suffix='.jpg', num_workers=8,
                                    test_name=test_name)
# learner = ConvLearner.pretrained(resnet34, data)


# ImageClassifierData:
@classmethod
def from_csv(cls, path, folder, csv_fname, bs=64, tfms=(None,None),
             val_idxs=None, suffix='', test_name=None, continuous=False,
             skip_header=True, num_workers=8):
    """ Read in images and their labels given a CSV file. """









help(fastai_osx.dataset.ImageClassifierData)


help(fastai_osx.dataset.ImageData)
