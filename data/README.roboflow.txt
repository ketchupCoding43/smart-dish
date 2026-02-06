
new-food-dataset - v1 2025-12-30 1:28pm
==============================

This dataset was exported via roboflow.com on December 30, 2025 at 7:59 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 126 images.
Food-item-detection are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Fit within)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, upside-down
* Randomly crop between 0 and 10 percent of the image
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -13 and +13 percent
* Salt and pepper noise was applied to 0.54 percent of pixels


