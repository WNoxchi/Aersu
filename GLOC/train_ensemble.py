# Wayne Nixalo - 2017-Dec-25 22:58
# GLOC Detector Ensemble Training Script

from fastai_lin.imports import *
from fastai_lin.conv_learner import *
from fastai_lin.model import *
from fastai_lin.torch_imports import *

# from utils.subfolder_val_loader import set_cv_idxs

import time
import os


# PATH = 'data/'
PATH = os.path.expanduser('~') + '/Aersu/GLOC/data/'
labels_csv = PATH + 'labels.csv'
val_idxs = [0]  # FastAI bug; need some val idxs

λr = 5e-3
wd = 1.25e-3


def get_data(size, bs=32, resize=False, test_name=None):
    tfms = tfms_from_model(arch, size, aug_tfms=transforms_side_on, max_zoom=1.2)
    data = ImageClassifierData.from_csv(PATH, 'train', labels_csv, bs=bs, tfms=tfms,
                                        val_idxs=val_idxs, suffix='jpg', num_workers=8,
                                        test_name=test_name)
    if resize:
        data.resize(int(size), 'tmp')
    return data

def train_loop(learner):
    t0 = time.time()

    print(f'TRAINING LOOP {model_name}: {learner.data.sz} - 1/10')
    learner.fit(lrs=λr, n_cycle=1, cycle_len=1)
    print(f'TRAINING LOOP {model_name}: {learner.data.sz} - 2/10')
    learner.fit(lrs=λr, n_cycle=3, cycle_len=1, wds=wd, use_wd_sched=True)
    print(f'TRAINING LOOP {model_name}: {learner.data.sz} - 5/10')
    learner.fit(lrs=λr, n_cycle=3, cycle_len=1, cycle_mult=2, wds=wd, use_wd_sched=True)

    t = time.time() - t0
    H = int(t / 3600)
    M = int((t - H*3600) / 60)
    S = t - H*3600 - M*60

    print(f'{model_name} TRAINING COMPLETE. TIME: {H}:{M:0=2d}:{S:.3f}\n')

def train_model(arch, model_name):
    data = get_data(100)
    learner = ConvLearner.pretrained(arch, data)
    train_loop(learner)

    data = get_data(200)
    learner.set_data(data)
    train_loop(learner)

    data = get_data(400)
    leraner.set_data(data)
    train_loop(learner)

    learner.save(model_name)

def main():

    #### RESNET 34 ####
    model_name = 'GLOC_RN34'
    arch = resnet34
    train_model(arch, model_name)

    #### RESNET 50 ####
    model_name = 'GLOC_RN50'
    arch = resnet50
    train_model(arch, model_name)

    #### DENSENET 121 ####
    model_name = 'GLOC_DN121'
    arch = dn121
    train_model(arch, model_name)

    #### DENSENET 169 ####
    model_name = 'GLOC_DN169'
    arch = dn169
    train_model(arch, model_name)

    #### RESNEXT 50 (32X4) ####
    model_name = 'GLOC_RNX50'
    arch = resnext50
    train_model(arch, model_name)

    #### RESNEXT 101 (32X4) ####
    model_name = 'GLOC_RNX101'
    arch = resnext101
    train_model(arch, model_name)

    #### RESNEXT 101 (64X4) ####
    model_name = 'GLOC_RNX101-64'
    arch = resnext101_64

    data = get_data(100)
    learner = ConvLearner.pretrained(arch, data)
    train_loop(learner)

    data = get_data(200)
    learner.set_data(data)
    train_loop(learner)

    data = get_data(400, bs = 26)
    leraner.set_data(data)
    train_loop(learner)

    learner.save(model_name)

    #### WIDERESNET 50 (24) ####
    model_name = 'GLOC_WRN50'
    arch = wrn
    train_model(arch, model_name)

    #### INCEPTION RESNET V2 ####
    model_name = 'GLOC_IRNV2'
    arch = inceptionresnet_2
    train_model(arch, model_name)

    #### INCEPTION V4 ####
    model_name = 'GLOC_IV4'
    arch = inception_4
    train_model(arch, model_name)

    #### VGG16BN (FASTAI) ####
    model_name = 'GLOC_VGG16_224'
    arch = vgg16
    data = get_data(100)
    learner = ConvLearner.pretrained(arch, data)
    train_loop(learner)

    data = get_data(224)
    learner.set_data(data)
    train_loop(learner)

    learner.save(model_name)

    #### RETINANET ####
    # TODO

if __name__ == "__main__":
    # main()

    #### PATH DEBUG:
    import os

    print(os.listdir(os.getcwd() + '/data/train/'))

    


















#
