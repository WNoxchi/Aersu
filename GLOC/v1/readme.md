# GLoC Detector version 1

--------------------------------------------------------------------------------

1. [Overview](#overview)
2. [Demo](#demo)
3. [Example Workflow](#examplework)
4. [Notebooks](#notebooks)


This is the first version of the GLoC Detector, written between December 2017 and February 2018 as a learning exercise in Image Classification and Object Detection. The system is a 2-Stage pipeline using a [Keras/TensorFlow RetinaNet](https://github.com/fizyr/keras-retinanet) (ResNet50 w/ [Focal Loss](https://arxiv.org/abs/1708.02002)) as its 1st-Stage detector, and a fastai/PyTorch ResNet34 as 2nd-Stage classifier.

## <a name="overview">Overview</a>

The complexity of this version arises from it being an iterative learning exercise with changing goals. After seeing a demo of the ['Boring Detector'](https://github.com/lexfridman/boring-detector), I wanted to be able to find a pilot in an image. Thinking about how to make this useful to the overall aim of the project, I decided to experiment with a 2-stage semi-supervised model. The first stage in 'automatic' mode crops out the bounding box of its highest-confidence detection. That cropped image is fed directly into the 2nd-stage classifier. In 'semi-supervised' mode, the top 5 detections are displayed and the operator can choose the best matching detection, or reject them and manually enter coordinates. The cropped images are saved to a folder and used as an 'interstage' training set for the classifier. The idea with this approach is to experiment with making the classifier more accurate by giving it less 'useless' information which it has to learn to ignore.

In early February I decided I needed to learn more about Computer Vision and Object Detection and began a [research deep-dive](http://forums.fast.ai/t/part-2-lesson-9-wiki/14028/230?u=borz). Fast.ai's [Deep Learning II](http://course.fast.ai/part2.html) course started around the same time, and by its end I decided on a full rewrite. This directory is a refactor of the old code.

## <a name="demo">Demo</a>

A video demo of the v1 GLoC Detector can be found by clicking the thumbnail below.

<p align="center"> <a target="link" href="https://www.youtube.com/embed/9x0SjXQ3F-A"><img src="https://img.youtube.com/vi/9x0SjXQ3F-A/0.jpg" width="80%" alt="GLoC v1 Dev Demo Video Link"></a></p>

If you want to run it on your own machine without going through all the steps to build the dataset:

1. Git clone the [fastai](https://github.com/fastai/fastai) and [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) repositories. The GLoC detector needs access to the fastai/fastai/ folder to work (you can either move that folder into demo/ or make a symlink (on a Posix OS) via: `ln -s PATH_TO_fastai/fastai/ fastai` in your terminal. The keras-retinanet repo is only needed for its install.

2. Create your conda `gloc` environment by typing `conda env create -f environment.yml` in this directory.

3. Go to the keras-retinanet/ folder you cloned, and with your `gloc` environment activated (`source activate gloc` on MacOS/Linux) type: `pip install .`. You should now be all setup.
	- **NOTE**: these steps install *a lot* of packages. You can try setting up a bare-bones environment and add packages when things don't work. The big items that are needed are: keras 2.1.5, keras-resnet 0.1.0,  tensorflow 1.8.0, [mss](http://python-mss.readthedocs.io/index.html) 3.1.2, opencv-python 3.4.0.12, numpy 1.14.3, torch 0.3.0.post4, ncurses 6.0. The keras-retinanet & fastai steps are still required, though.

4. Go to the demo/ folder. A dummy dataset is already there to initialize the models. To download and intialize the models (~200MB) run `python dummy_resnet.py` then `python get_retinanet.py`. To run the demo type: `python display_demo.py`. The detector will begin running on the top-left Press `q` with the display window selected to quit - the program will exit after the next update.

## <a name="examplework">Example Workflow</a>

In theory, and ideally, the full workflow of the GLoC Detector consists of 4 main steps, with an optional two-part semi-supervised training step.

1. Data Collection & Formatting.

2. Model Development and Training.

3a. Optional: Building an Interstage Dataset

3b. Optional: Semi-Supervised Finetuning

4. Running the GLoC Detector

Development reality is much messier. Each step was planned, implemented, and updated as the project went along. Some steps were given too much attention and turned out to be trivial, others were brushed over but presented significant new technical challenges. During development, new ideas and improvements to existing ideas contributed to complexity. I went in knowing there were details I couldn't know until I met them, and I accepted a level of roughness with this first version.

This is why, for example, the 1st step is implemented by 5 files in version 1, but in the new rewrite just 2: ge- data, format-data.

I don't expect anyone to fully go through the v1 workflow because (aside from being outdated) of a number of technical issues. Chiefly: the keras-retinanet repository I relied on for the 1st-Stage Pilot Detector has had some significant changes to both its pretrained RetinaNet models and its API. Fixing/debugging the code and finding specific dependencies took the better part of a night to make the `display_demo.py` script work out-of-the-box. The new GLoC Detector will be a single End-to-End fastai/PyTorch object detector.

Despite that, a v1 workflow would look something like this (and this is the summarized version of how I build the project):

### Data Collection and Formatting

Setup up Python 3 conda environment for fastai/PyTorch and Keras/TensorFlow use. Make sure other packages ([mss](http://python-mss.readthedocs.io/index.html) for screen shots, etc) installed. Use the YouTube links from `urls.txt` and ensure the `bbox` tuple (x1, y1, x2, y2) in `datagetter.py` matches up with the video's position on screen. A MacOS Retina screen will needs different values than a normal one (I think 2x).

1. run `python datagetter.py` in your terminal.
	- Hit `SPACE` to save an image with a positive label (pilot is passed out), and any other key (`M` works fine) for a negative label. Hit `ENTER` to quit. `datagetter.py` will save the cropped screenshots to the data/ folder, automatically named in the order they were saved starting from 1 + the highest numbered .jpg file in the folder. It'll also create a `labels_START-END.csv` file of image-id and GLoC label ('1' is positive).
	
2. run `python subfolderize.py` in your terminal.
	- Images saved to data/ will be put into subfolders based on the label CSVs.
	- This can be undone by running `python moveallout.py`.

3. run `python combine_labels_script.py` in your terminal.
	- This combines the multiple CSV label files into a single master CSV which'll be used by the neural net models.
	
4. run `python name_corrector.py` in your terminal.
	- This converts file ids from `XX` to `0000XX`.

5. run `python directorize_fnames.py` in your terminal.
	- This converts the file ids from `0000XX` to `subdirectory/0000XX`
	- This allows for data-leakage-free validation set creation through the CSV, without having to move any files.
	
### Model Development and Training

The notebooks/ folder contains all notebooks created during development and testing. The final model I ended up using is trained using the steps in notebook/Ensemble-Prelim.ipynb -- in the **Paths and Imports** through **ResNet34** sections. Due to an issue with the tqdm library at the time, status bars printed out at each update in Jupyter notebooks, creating the horrendous printouts you'll see there. They can be ignored. The final regimen uses progressive resizing ([a technique fast.ai used to reach top rankings in the Stanford DAWNBench competition](http://www.fast.ai/2018/04/30/dawnbench-fastai/)). You are done when you see the `learner.save('RN34_400_WD_λ0-529_00')` cell followed by **ResNet50**.

The actual classifiers did not do well and suffered from overfitting. This was part of the reason I decided to focus on Computer Vision research for a while before coming back to the problem. Learning rates were chosen using fastai's Learning Rate Finder which implements an idea in Leslie Smith's [Cyclical Learning Rates paper](https://arxiv.org/abs/1506.01186).

### Interstage Dataset & Semi-Supervised Finetuning

`cropper.py` is used to built the interstage dataset. This looks for the file-id last numbered-CSV file in the interstage data folder, or creates the folder and starts from zero if none exist. Using the file-id (+ 1) as a starting index, `cropper.py` begins iterating through every image specified by the labels.csv master CSV. At each iteration, the image is displayed along with 5 colored bounding boxes (which I made sure to be equidistant to each other in color via `utils.common.c_shift` to make it easier to tell them apart) and a printout the the index and coordinates of each box. The operator then enters the index of the best match, or if none of them are good enough, enters `0` to switch to manual mode and specify the (x1, y1, x2, y2) coordinates of the 'correct' detection. The OpenCV window shows the coordinate position of the cursor. The operator can quit by entering `q` in the terminal, at which point the array of bounding boxes is saved into an `interstage_labels_START-END.csv` file. `combine_labels_script.py` can be run (specifying the correct in/out names) to create a 'master' interstage-labels file.

The 2nd-Stage classifier can be trained on this new dataset the same way it was initially trained. `cropper.py` saves a center-crop-on-black of every image it goes through based on the chosen bounding box coordinates to the interstage dataset.

### Running the GLoC Detector

Just as in the Demo section above, `display_demo.py` is run and the GLoC Detector begins real-time work on the coordinates of the screen specified. The models are loaded, and at each iteration a screenshot is taken, sent to the 1st-Stage Detector which crops its top detection, and that crop is sent to the 2nd-Stage Classifier to determine if it's of a person suffering GLoC or not. Finally the bounding box of the detection is drawn on the original image along with the [negative, positive] GLoC confidence scores, and displayed in a window.

If `show_time = True` in `display_demo.py` the time taken for each step (time to initialize tensorflow; time for stage 1; time for stage 2) is displayed in the terminal. Pressing `q` with the display window selected will quit after the next update.

*And that's the v1 GLoC Detector*. My application-based crash course in Computer Vision, applied Deep Learning, Python, Pandas, runtime optimization, OpenCV, and a lot more. Personally, I'm looking forward to making v2 a high-speed high-accuracy lean system.

-----------------------------------------------------

## <a name="notebooks">Aside: the Notebooks</a>

I wanted to document my progress as much as possible - nearly everything I encountered was new. The downside of this was a lot of notebooks that weren't very easy to organize in the middle of work. Also, all of my actual model development (and a lot of code testing) was done in these notebooks. In broad strokes:

- [Ensemble-Prelim.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/Ensemble-Prelim.ipynb): my first attempt at creating a giant ensemble. I did it, but it didn't help much. The models were too similar and likely suffering from similar causes of failure. I also had no where near the resources to run them live.

- [Ensemble-Final.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/Ensemble-Final.ipynb): aborted planned second attempt. Aborted after seeing there wasn't a clear way to improve the first attempt, and transitioning to a 2-Stage detector-classifier was a higher priority.

- the 3 `debugging-...` notebooks [1](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/debugging-and-testing.ipynb)-[2](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/debugging-demo-display.ipynb)-[3](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/debugging-dummy-trainingset-error.ipynb): exactly what they suggest. Mostly learning about Python, Pandas, and the fastai library. Also testing out / fixing pieces of code before deploying them in the data-formatting scripts. Learning how to quickly manipulate tabular and image data was actually a dominant part of this project.

- [DevNotes_GLOC.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/DevNotes_GLOC.ipynb): an attempt at live documentation late in the project. At this point I was preparing to scrap the Keras 1st stage (which was meant as a prototype to test Object Detection anyway) for a PyTorch SSD-based model. A little [back & forth on twitter](https://twitter.com/jeremyphoward/status/962439072739311616) about this, and trouble-shooting the keras-retinanet API.

- [Initial-Model-RN34.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/Initial-Model-RN34.ipynb): my initial main thrust into the problem, starting off by using a [fastai DL1 code along](https://github.com/WNoxchi/Kaukasos/blob/master/FADL1/L3CA_lesson2-image-models.ipynb) I did as a reference. A lot of attacking the problem and lots of learning. There's also a silly picture of me that proved my model was not generalizing.

- finetune-retinanet-dev [1](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet-dev.ipynb)-[2](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet-dev2.ipynb)-[3](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet-dev3.ipynb)-[4](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet-dev4.ipynb): my initial attempts at trying to finetune the MS-COCO pretrained RetinaNet model on my data. This is where I began learning about their API, and really got hit in the face by the need for *entirely different versions of a model for inference & training in Keras due to static computation graphs*.

- finetune-retinanet_attempt [01](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet_attempt01.ipynb)-[02](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/finetune-retinanet_attempt02.ipynb): I tried to finetune the RetinaNet model and failed badly. After this I experimented with just using the stock model (since it was already trained to detect "person" - which neatly intersected my trying to detect "pilot").

- image_overlay-development.ipynb(https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/image_overlay-development.ipynb): Learning how to overlap a cropped image over another image; how to create a center-crop on black. It was very late and I had too much fun with this..

- [keras-retinanet-example.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/keras-retinanet-example.ipynb): in order to get a feel for how to use a different framework, API, and pretrained model, I do a code along with my own images. **Note**: there are a lot of images of Chamath Palihapitiya. I was watching a [very good talk of his](https://www.youtube.com/watch?v=PMotykw0SIk) when I started the GLoC project, so I endup using a screengrab of him to test a lot of Computer Vision stuff.

- [k-retinanet-prelim-pilot.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/k-retinanet-prelim-pilot.ipynb): I do more work applying RetinaNet to my dataset. Here I get the idea for Semi-Supervised learning and decide to create an interstage dataset, by using the model to help me quickly hand-annotate the 7,637 images in my dataset. That was a long day.

- loading-from_arrays- [1](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/loading-from_arrays-1.ipynb)-[2](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/loading-from_arrays-2.ipynb): in fast.ai we learned a lot of Deep Learning from a Data-Analytics / Machine Learning perspective (static datasets like Kaggle, ImageNet, etc). I envisioned this project as being a real-time possibly embedded system that could react in milliseconds if a pilot passed out - if done as a proper company. So I wanted to see how I could quickly load individual images and use the power of the fast.ai library on them. I was also experimenting with moving away from using the `.from_paths` ModelData constructor. I didn't have much experience with CSVs, so I thought loading NumPy arrays would work well (I ultimately found the `.from_csv` method to be an all-purpose superpower of the fastai library for training) -- though I used `.from_arrays` for the real-time 'online' operation you see in the demos.

- [retinanet-k-coco-label-function.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/retinanet-k-coco-label-function.ipynb): I had a *lot* of trouble working with the Keras RetinaNet model on my own terms. I learned a lot about OOP by diving into and pulling apart the `CocoGenerator` class it used for its dataloading. Here is also my first introduction to the JSON for storing multi-object annotations and labels.

- [retinanet-k-visualizing-labels-coco.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/retinanet-k-visualizing-labels-coco.ipynb): as the title says. I was working towards displaying the top-5 detections for use in building the interstage dataset.

- [scratch.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/scratch.ipynb): testing `glob` vs `iglob` times. A few looks at my CSVs with Pandas.

- [stitcher-dev-testing.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/stitcher-dev-testing.ipynb): "Working on how exactly I'll read in data from numbered CSV files, and concatenate them into a single main CSV file."

- [v0.1.1-RN34-valfolders.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/v0.1.1-RN34-valfolders.ipynb): one of my first forays into the problem, during 'v0': my first dataset. I initially went with a 'bigger is better' mentality and got around 35,310 images in my dataset. I tried using progressive resizing and weight-decay with Adam. I figured that the images were too self-similar and the dataset was way too imbalanced with negative examples. I rebuilt the dataset after this to the 7,637-image dataset I have now.

- [v0.2.0-RN34-newdataset.ipynb](https://github.com/WNoxchi/Aersu/blob/master/GLOC/v1/notebooks/v0.2.0-RN34-newdataset.ipynb): my next work with the rebuilt dataset. I discovered overfitting by trying to use silly pictures I took of myself as tiny out-of-data test set.
