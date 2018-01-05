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
