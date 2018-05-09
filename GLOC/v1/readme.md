# GLoC Detector version 1

This is the first version of the GLoC Detector. This version uses a 2-stage pipeline, and was written when I was new to Object Detection. This version of the project was very much a learning and exploration exercise. The intro to OpenCV work which motivated me to start this can be found in the `WNoxchi/DoomPy/AutoPy/` repository.

Stage 1 is a pilot detector using a Keras/TensorFlow-based RetinaNet (ResNet + Focal Loss; pretrained on MS-COCO) to localize a pilot in an image or video stream. Stage 2 is a fastAI/PyTorch-based (ResNet; pretrained on ImageNet) GLoC classifier. Stage 1 crops a pilot out of an image and sends it to Stage 2. Stage 2 then classifies the image as a pilot undergoing GLoC or not.

Since version 1 I’ve gone through fast.ai’s Deep Learning II course and have opted for a near-full rewrite into an end-to-end ‘conscious-person’ detector. You can find that (as of 9 May ’18 in-progress) work in the directory above this one: `WNoxchi/Aersu/GLOC/`.

## Usage Notes:

### Before you dive in:

You may wonder why the steps below are so convoluted and drawn out. This is because this was a learned-work-in-progress. As I progressed through the project I realized some ideas weren’t worth keeping, or better ideas became apparent (for example: the division into video-link-based subfolders made it easier to create validation sets without data-leakage). All of this is corrected in the version 2 rewrite.

I don’t have a link to download the dataset. You can rebuild the dataset yourself by running each video link in the `urls.txt` file, checking that the `bbox` argument in `save_screen_box` in `datagetter.py` lines up with the youtube video, and following the instructions in the terminal after running `datagetter.py`. It is tedious and imperfect, but that’s how I got the data.

### Building a new G-LoC Dataset:

**Getting the data**:
- `datagetter.py`

**Correct CSV filenames to include subfolder path**:

**Merge all CSVs into a single `labels.csv file`**:
- ‘combine_labels_script.py’

**Correct file ids from `XX` $\rightarrow$ `0000XX`**:
- `name_corrector.py`

### Example Workflow:

0. Ensure `aersu` conda environment (Py3) is setup as per `aersu-env-reqs.txt`. The fast.ai library will need to be installed separately (a quick way to do this is to follow the install instructions at https://github.com/fastai/fastai, clone or rename your ‘fastai’ environment as ‘aersu’, then conda -env update with the GLoC `environment.yml` file. Sorry for the inconvenience. All fastai/pytorch code in the project will require the presence of or a symlink to the fastai library (fastai/fastai/).

1. run `python datagetter.py` in your terminal.
	- Hit `SPACE` to save an image with a positive label, and any other key (`M` works fine) for a negative label. Hit `ENTER` to quit. `datagetter.py` will save the cropped screenshots to the /data/ folder, named by index (starting from the highest-index .jpg file in data/. It’ll also save a `labels_START-END.csv` file with matching indices.

2. run `python subfolderize.py` in your terminal.
	- Images saved to data/ will be put into subfolders based on the label CSVs. The assumption is that each CSV references images taken from different videos (hence different people to prevent data leakage).
	- This can be undone by running `python moveallout.py`.

3. run `python combine_labels_script.py` in your terminal.
	- This combines the multiple CSV label files into a single master CSV which will be used by the neural net models.

4. run `python name_corrector.py` in your terminal.
	- This converts the file ids from `XX` --> `000XX`

5. run `python directorize_fnames.py` in your terminal.
	- This converts the file ids from `000XX` --> `subdirectory/000XX`

### Running the GLoC Detector:

- run `python display_demo.py`.
**NOTE**: GLoC v1 just uses the pretrained RetinaNet model as a detector. You can download the pretrained MS COCO RetinaNet model [from this link](https://github.com/fizyr/keras-retinanet/releases/download/0.2/resnet50_coco_best_v2.0.3.h5) (from this [github repo](https://github.com/fizyr/keras-retinanet)). Make sure it is saved as `data/retinanet-model/resnet50_coco_best_v1.2.2.h5` - as that’s where `display_demo.py` will look to load the model.
