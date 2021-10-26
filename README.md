# sign_understanding

To do sign understanding from a single RGB images, this involves two main steps:
1. Detect objects in the scene (e.g., texts, arrows, symbols, etc.), and
2. Group the objects together.

This repository contains the code to do the above two steps.

Once these are done, we can use existing OCR approaches (e.g., easyOCR or Tesseract) to read the text and train other machine learning technique to identify arrow directions and symbols. Once these are all available, we can identify which texts and symbols belong to which direction. 

If combined with 3D, we could also use homography to rectify the signs, making the direction easier to parse.

This repo contains 2 methods to tackle the above two steps.

## 1. Rule-based method
The rule-based method is in the file `sign_detection_rulebased.ipynb`

It doesn't use machine learning to group the signs together.
Its main idea is based on the observations that
1. An arrow can be assigned to many texts/symbols, but text/symbols cannot be assigned to multiple arrows 
2. Object in the same group (arrow/text/symbols) usually belong to a sign/block with the same background color. And if there are many blocks sharing the same signboard with the same background color, then the objects in each block usually have smaller distance to each other then the distance to objects from other blocks.

Note that these observations are not always true, but roughly correct for many signs.
We designed the algorithm based on the two observations, which lead to a rule-based algorithm which is useful for signs in stations.

The algorithm works by first training Detectron2 object detector to detect the objects.
Then, the grouping is performed as
1. Using each arrow as the initial state of each group (i.e., the number of groups = the number of arrows)
2. If a text or a symbol has the same background color as one and only one arrow and such background is the same connected component (defined as a group of adjacent pixels with similar colors), then assign the text/symbol to the arrow's group.
3. If there are multiple arrows with the same background color in the same connected component, then greedily assign text and symbol to the an existing group based on the distance to one of the existing group member.

At this moment, this approach is more understandable and and seems to make more sense, but the learning-based method below achieves better clustering accuracy.

## 2. Learning-based method
The code is in the file `sign_detection_learningbased.ipynb`.

Basically, we train a set-transformer network to output an affinity matrix, where the estimated affinity matrix is penalized to cluster the objects (text/arrows/symbols) correctly.
Spectral clustering is then performed on the estimated affinity matrix to obtain the clusters.
Note that, unlike the Rule-based method, we do not enforce any constraint, so a single group may have multiple arrows. Improving this point may lead to a better result.

# Setup environment

Setup dependencies by the following command.

```
pip install -r requirements.txt
```

Install Detectron2 by following the [guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (v0.5 is tested).

# Preprocessing
Before running the algorithms, we need to preprocess the data.
1. Annotate images by using [CVAT](https://github.com/openvinotoolkit/cvat) and download data. The annotations should be in COCO format. Since CVAT does not allow downloading all annotations from all tasks at once, we need to download them one by one. Put the data (`annotations` and `images` from the downloaded zip) in the a folder `dataset/raw`. Note that the default CVAT annotation file name is "instances_default.json", and you may rename annotation json files to a unique name in `dataset/raw`. Example directory structure is as follows.
```
dataset
|
|--- raw
     |--- annotations
     |       instances_default1.json
     |       instances_default2.json
     |       instances_default3.json
     |
     |--- images
          |--- image_dataset_directory_1
               |--- image1.jpg
               |--- image2.jpg
          |--- image_dataset_directory_2
               |--- image1.jpg
          |--- image_dataset_directory_3
               |--- image1.jpg
               |--- image2.jpg
               |--- image3.jpg
```
2. Merge the annotation into a single file using `merge_json.ipynb`. More explanation inside the notebook, and change paths as needed.
3. Then, run the `parse_annotations.ipyb` to convert the data into a format processable by the Learning-based method. Note that this data format is also used for evaluating the Rule-based method.

# Dependencies
- [Cython](https://cython.org) (Apache-2.0 License)
- [detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
- [easyocr](https://github.com/JaidedAI/EasyOCR) (Apache-2.0 License)
- [editdistance](https://github.com/roy-ht/editdistance) (MIT License)
- [imageio](https://github.com/imageio/imageio) (BSD-2-Clause License)
- [ipdb](https://github.com/gotcha/ipdb) (BSD-3-Clause License)
- [IPython](https://github.com/ipython/ipython) (BSD-3-Clause License)
- [matplotlib](https://github.com/matplotlib/matplotlib) (matplotlib License)
- [numpy](https://github.com/numpy/numpy) (BSD-3-Clause License)
- [opencv-python](https://github.com/opencv/opencv-python) (MIT License)
- [scikit-image](https://github.com/scikit-image/scikit-image) (BSD-3-Clause License)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause License)
- [scipy](https://github.com/scipy/scipy) (BSD-3-Clause License)
- [Shapely](https://github.com/Toblerity/Shapely) (BSD-3-Clause License)
- [tensorboardX](https://github.com/lanpa/tensorboardX) (MIT License)
- [torch](https://github.com/pytorch/pytorch) (BSD-3-Clause License)
- [torchvision](https://github.com/pytorch/vision) (BSD-3-Clause License)
- [tqdm](https://github.com/tqdm/tqdm) (MIT License / Mozilla Public Licence (MPL) v. 2.0 License)

# License

[MIT License](LICENSE)