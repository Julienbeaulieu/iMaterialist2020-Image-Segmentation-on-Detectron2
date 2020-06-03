# iMaterialist 2020 Kaggle Competition in Detectron2

In this competition we are tasked to do instance segmentation with attribute localization (recognize one or multiple attributes for the instances) on a fashion and apparel dataset. [Here is the link to competition](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview). 

<p><img src="https://julienbeaulieu.github.io/public/imaterialist/imaterialist-dataset.png" /></p>

## Model and Training

To solve the challenging problems entailed in this task we use and extend Detectron2’s MaskRCNN architecture and added a new attribute head as shown in orange below.  

<p><img src="https://julienbeaulieu.github.io/public/imaterialist/attribute-MaskRCNN%20model.png" /></p>

-	In prior steps in the MaskRCNN architecture we leverage a ResNet-50 with a feature pyramid network (FPN) as backbone. 
-	The input image is resized to 1300 of the longer edge to feed the network. 
-	Random horizontal flipping was applied during the training. 
-	The model was trained on top of pre-trained COCO dataset weights for 300,000 iterations.

## Kaggle Submission

The submission to Kaggle required specific encoding (run length encoding - RLE) for all the predicted masks in order to reduce the size of the submitted file. This posed a number of challenges since RLE is not standardized amongst COCO, Detectron2 and Kaggle. Also, Kaggle required that each pixel of the masks do not overlap, so mask refining was required.

## Evaluation

Submissions are evaluated on the mean average precision at two different thresholds.

1. IoU: intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

<p align="center"><img src="https://latex.codecogs.com/gif.latex?IoU(A,B) = \frac{|A\cap B|}{|A\cup B|} " /></p>

2. F1: f1 score between a set of predicted attributes and a set of true attributes of one segmentation mask

The metric sweeps over a range of IoU thresholds and F1 thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at an IoU threshold of 0.5 and an F1 threshold of 0.5, a predicted object is considered a "hit" if it satisfies the following conditions:

1.	Its intersection over union with a ground truth object is greater than 0.5
2.	If the ground truth object has attributes, the f1 scores of predicted attributes and ground-truth attributes is greater than 0.5.
At each threshold pair, t=(ti, tf), a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

<p align="center"><img src="https://latex.codecogs.com/gif.latex? \frac{TP(ti,tf) }{TP(ti,tf)+FP(ti,tf)} " /></p> 

## Category and Attributes Analysis 

There are 46 apparel categories and 294 attributes presented in the Fashionpedia dataset. On average, each image was annotated with 7.3 instances, 5.4 categories, and 16.7 attributes. Of all the masks with categories and attributes, each mask has 3.7 attributes on average (max 14 attributes).

## Docker 

A Docker image is available at https://hub.docker.com/r/cvnnig/detectron2.

## WIP

This repo is still being cleaned and organized.

## Authors

Julien Beaulieu, Yang Ding