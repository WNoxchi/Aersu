# WNixalo 2018-May-09 20:13
# Script to download the pretrained RetinaNet (ResNet with Focal Loss, ie:
# paramatarized scaled loss) from https://github.com/fizyr/keras-retinanet

from pathlib import Path
from urllib.request import urlretrieve
from os.path import exists
from os import makedirs

def main():
    d_link = 'https://github.com/fizyr/keras-retinanet/releases/download/0.2/resnet50_coco_best_v2.0.3.h5'
    retnet_path = Path('data/retinanet-model/')
    # fname = 'resnet50_coco_best_v2.0.3.h5'
    fname = 'resnet50_coco_best.h5'
    # NOTE: the authors may update their pretrained model, so to ensure this'll
    #       work with `display_demo.py` out-of-the-box in the future, I have it 
    #       set to this name.

    if not exists(retnet_path/fname):
        makedirs(retnet_path, exist_ok=True)
        print(f'Downloading from {d_link}')
        urlretrieve(d_link, retnet_path/fname)
        if exists(retnet_path/fname):
            print(f'Finished: {str(retnet_path/fname)}')
        else:
            print(f'Something went wrong.')

    else:
        print(f'File already exists: {str(retnet_path/fname)}')

if __name__ == '__main__':
    main()
