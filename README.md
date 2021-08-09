# Current work in progress
# Yolov4-tf2-SORT
- A integration of Yolov4-tf2, SORT, RFID detections for mice tracking in home-cage environment
- Original inspiration and prototype by [Braeden Jury](https://github.com/ubcbraincircuits/NaturalMouseTracker)
- A [raspberry-pi based system](https://github.com/tf4ong/tracker_rpi) for offline tracking of mice


## Analytical Codes and Modules
##### Yolov4-tensorflow2 
yolov3-tf2:forked from [the AI guys GitHub Page](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
- A tensorflow2 implementation of yolov4 which is easy to install and use
- Can be cloned and directly used on Anaconda or on Google Colab with GPU turned on 
- Simple object recognition module for detecting mice
- Orginal implmentation in darknet can be found at [AlexeyAB GitHub Page](https://github.com/AlexeyAB/darknet)
##### SORT 
SORT: forcked from [abewly GitHub Page](https://github.com/abewley/sort)
- A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
- Implements a visual multiple object tracking framework based on 
rudimentary data association and state estimation techniques. 
- Produces object identities on the fly and tracks them 
- Initiatially described in [this paper](https://arxiv.org/abs/1602.00763)
- Greatly depends on detection quality
- Handles detection lost during occlusions and Yolov3 failures
##### RFID_Matching 
Customed writted script for track reassocation and RFID assignment
###### SORT Track Identity Reassociation: A Euclidean Distance Method
SORT was orignally designed for tracking objects moving in and out of frames at relatively uniform speed.
Mice movements are often fast changing, therefore SORT often produces new identities for the same mouse.
Taking advantage of known number of mice detected in the previous frames and that a new mouse can only enter at the 
designated location, we can therfore reassign new false positive identities to real identities generated. Here, 
a centroid tracking algorithm based on Euclidean distances is employed. A tutorial of centroid tracking can be found 
[here](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)
###### RFID Identification and Verification
o


![](Sample_RFID.gif)
