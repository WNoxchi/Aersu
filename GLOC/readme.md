# GLoC Detector

[G-Induced Loss of Consciousness](https://en.wikipedia.org/wiki/G-LOC)

Wayne H Nixalo

## Overview

**[v1/](https://github.com/WNoxchi/Aersu/tree/master/GLOC/v1)** : version 1
  - [demo video](https://www.youtube.com/embed/9x0SjXQ3F-A)

This directory is home to the current v2 GLoC Detector under active development. See [`v1/`](https://github.com/WNoxchi/Aersu/tree/master/GLOC/v1) for a working demo of the previous complete version. 

V1 uses a 2-Stage pipeline with a Keras/TensorFow 1st-stage Detector feeding a 2nd-Stage FastAI/PyTorch Classifier. V2 will be a single End-to-End FastAI/PyTorch Network using a Regressor Detection head and a Classification head sharing a common convolutional backbone.

---

## Dataset

Unpacking and formatting the raw dataset:

- Run: `dataset_formatter.py`

Creating a raw dataset:

- Load a video to your screen. Run `datagetter.py` while video plays (aligning video or editting dimensions in script as necessary).

If you want to move the dataset from its subfolders into the data directory:

- Run `moveallout.py`
