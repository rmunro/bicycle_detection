# bicycle_detection
Practical example from Human-in-the-Loop Machine Learning book

## Getting started

To run:

python bicycle_detection.py

This will open a HTML window that will allow you to annotate whether a 
given image contains a bicycle or not. 

The goal is to annotate enough data to train a system that is more accurate 
than the current state-of-art results on this dataset (about 0.85 F-score).


## Warning: large memory and network footprints

This repo will take up approximately 1GB of space locally.

For every image that you annotate, it will take an additional 10KB. 
So, 50K images will take up about 500MB extra disk space. 

The images are _not_ stored locally: only the features (hence only 10KB per image).
However, the full images are temporarily downloaded to extract the features, so 
there will be a large amount of data over your internet connection.
 


## Data

The data is approximately 1 million images taken from the Open Images dataset.

## Problem being addressed

Transportation researchers want to estimate the number of people who use 
bicycles on certain streets.

- “I want to collect information about how often people are cycling down a street”
- “I want to capture this information from thousands of cameras and 
I don’t have the budget to do this manually”
- “I want my model to be as accurate as possible”

## Annotation strategy

The interface is optimized for rapid annotation, 
showing a continual list of images that you can rapidly annotate. 

When enough images have been annotated to start building models, 
the current accuracy will be shown in the window and the system will also
start using uncertainty sampling and model-based outliers to sample 
images that are most likely to help improve the overall accuracy of your model.

The machine learning model combines a ResNext50 model trained on ImageNet data
and a Faster R-CNN model trained on COCO data.


## Potential extensions

There are many different components in this architecture that could be extended or replaced. 
After playing around with the interface and looking at the results, think about what you might replace/change first.
(Numbers refer to book/chapter sections, but you don't need the book to experiment with this code.)

### Annotation Interface

- Batch Annotations (See 11.2.1): We could speed up annotation by allowing batch annotation interfaces. For example, an interface with 10 or so images where the annotator only has to select those with bicycles in them.
- Bounding-box annotation (See 11.5.2): For images containing bicycles if the model cannot predict that bicycle correctly, the annotator can highlight the bicycle. That image can be used as a cropped training data example to help guide the model on similar examples.

### Annotation Quality Control

- Intra-annotator agreement (See 8.2): Domain experts often under-estimate their own consistency, so it might help to repeat some of the items for the same food safety expert. This will help measure whether this is the case.
- Predicting Errors (See 9.2.3): Build a model to explicitly predict where the expert is most likely to make errors, based on Ground-Truth Data, Inter/Intra-Annotator Agreement and/or the amount of time spent on each report (assuming that more time is spent on more complicated tasks). Then, use this model to flag where errors might occur and ask the expert to pay more attention and/or give those items to more people. 

### Machine Learning Architecture

- Object Detection: If the images can be automatically cropped and/or zoomed in on the parts where the bicycles are predicted to be located, then this could improve the speed and accuracy of the annotation process.
- Continuous/Contiguous task. The task definition implies that the transportation manager is interested in the number of bicycles, not simply whether or not one or more occur. So, the model could be more useful if it was a continuous or contiguous task predicting the exact amount. Note that the annotation will be slower and the quality control more difficult to implement.

### Active Learning

- Ensemble-based Sampling (See 3.4): Maintain multiple models and track the uncertainty of a prediction across all the models, sampling items with the highest average uncertainty and/or the highest variation in predictions. 
- Representation Sampling (See 4.4) We are using pre-trained models from ImageNet and COCO, but we are applying the model to Open Images. So, we could use Representative Sampling to find the images most like Open Images, relative to the other sources, as errors are more likely to occur there. 

