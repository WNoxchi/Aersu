# Wayne H Nixalo - 2018-Jan-02 19:59

# fizry/keras-retinanet/keras_retinanet/preprocessing/coco.py
# cocodataset/cocoapi/PythonAPI/pycocotools/coco.py

# load model
from keras_retinanet.resnet import custom_objects
model = keras.models.load_model('data/retinanet-model/resnet50_coco_best_v1.2.2.h5',
                                custom_objects=custom_objects)

# get predictions on image
_,_, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
label = np.random.choice(predicted_labels)


# Replaces OOP CocoGenerator
classes, labels = load_classes_and_labels()

# IMPORTANT PART
caption = label_to_name(label)



# Functional Implementation of COCO and CocoGenerator classes for labels
def load_classes_and_labels(data_dir='data/COCO', set_name='val2017'):
    from collections import defaultdict
    import json
    import time
    import os

    ### class COCO
    # __init__(self, annotation_file=None)
    dataset, anns, cats, imgs = dict(), dict(), dict(), dict()

    annotation_file = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
    # from: class CocoGenerator.__init__: self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))

    if not annotation_file == None:
        print('loading annotations...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        assert type(dataset)==dict, f'annotation file format {type(dataset)} not supported'
        print(f'Done (t={time.time()-tic:0.2f}s)')

    # createIndex(self)
    print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

        if 'categories' in dataset:
            for ann in dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img

    if 'categories' in dataset:
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    print('index created!')

    # getCatIDs
    catNms=[]; supNms=[]; catIds=[]     # just to preserve some semblance of the OOP method form

    catNms = catNms if _isArrayLike(catNms) else [catNms]
    supNms = supNms if _isArrayLike(supNms) else [supNms]
    catIds = catIds if _isArrayLike(catIds) else [catIds]

    if len(catNms) == len(supNms) == len(catIds) == 0:
        cats_ = dataset['categories']
    else:
        cats_ = dataset['categories']
        cats_ = cats_ if len(catNms) == 0 else [cat for cat in cats_ if cat['name']          in catNms]
        cats_ = cats_ if len(supNms) == 0 else [cat for cat in cats_ if cat['supercategory'] in supNms]
        cats_ = cats_ if len(catIds) == 0 else [cat for cat in cats_ if cat['id']            in catIds]
    ids = [cat['id'] for cat in cats_]

    # loadCats(self, ids=[])
    if _isArrayLike(ids):
        categories = [cats[id] for id in ids]
    elif type(ids) == int:
        categories = [cats[ids]]

    ### class CocoGenerator(Generator)
    # __init__(self, data_dir, set_name, image_data_generator, *args, **kwargs)
    # load_classes(self)
    categories.sort(key=lambda x: x['id'])

    classes             = {}
    coco_labels         = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    return classes, labels

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

# name_to_label(self,name)
def name_to_label(name, classes):
    return classes[name]

# label_to_name(self, label)
def label_to_name(label, labels):
    return labels[label]

# classes, labels = load_classes_and_labels()
