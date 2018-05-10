# WNixalo 2018-May-09 19:31
# Saves a model for a pretrained ResNet34 classifier. No training is done.

from fastai.conv_learner import *

def main():

    PATH = Path('data')
    IMG_PATH = PATH/'train'
    CSV_PATH = PATH/'labels.csv'

    sz   = 224
    bs   = 16
    tfms = tfms_from_model(resnet34, sz)
    val_idxs = [0]

    model_data = ImageClassifierData.from_csv(PATH, IMG_PATH, csv_fname=CSV_PATH,
                                bs=bs, tfms=tfms, suffix='.jpg', val_idxs=val_idxs)
    learner = ConvLearner.pretrained(resnet34, model_data)

    learner.save('dummy_resnet34_classifier')

if __name__ == '__main__':
    main()












# model_data = ImageClassifierData.from_arrays(PATH, trn, val, bs=16)
# models = ConvnetBuilder(resnet34, 2, 0, 0, ps=None, xtra_fc=None, xtra_cut= 0,
#                         custom_head=None, precompute=False, pretrained=True)
# learner = ConvLearner(model_data, models)
