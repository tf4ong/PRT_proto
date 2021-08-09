# Yolov4-tf2-SORT
- A integration of Yolov4-tf2, SORT, RFID detections tracking mice
- To analyze data generated from [PSYCO recording software on raspnberry pi](https://github.com/tf4ong/tracker_rpi)
- Data analyzed offline using a pc/cloud notebook with cuda capable GPU


## Analytical Codes and Modules
#### Yolov4-tensorflow2 
yolov3-tf2:forked from [the AI guys GitHub Page](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
- A tensorflow2 implementation of yolov4 which is easy to install and use
- Can be cloned and directly used on Anaconda or on Google Colab with GPU turned on 
- Simple object recognition module for detecting mice
- Orginal implmentation in darknet can be found at [AlexeyAB GitHub Page](https://github.com/AlexeyAB/darknet)
- Three weight included as of the current release
- Weights are adviced to be trained on the original darknet implementation
- The AI guys has provided an excellent [tutorial](https://www.youtube.com/watch?v=mmj3nxGT2YQ) to train using google colab.
#### SORT 
SORT: forcked from [abewly GitHub Page](https://github.com/abewley/sort)
- A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
- Implements a visual multiple object tracking framework based on 
rudimentary data association and state estimation techniques. 
- Produces object identities on the fly and tracks them 
- Initiatially described in [this paper](https://arxiv.org/abs/1602.00763)
- Greatly depends on detection quality
- Maintains tracking throughout occlusions and clustering
- Also used to generated Kalmen filter predictions of mice locations when occuluded/clustered
### SORT Track Identity Reassociation: A Euclidean Distance Method
SORT was orignally designed for tracking objects moving in and out of frames at relatively uniform speed.
Mice movements are often fast changing, therefore SORT often produces new identities for the same mouse.
### Home Cage Tracking
Taking advantage of known number of mice detected in the previous frames and that a new mouse can only enter/exist at a designated location (Entrance), we can therfore reassign new false positive identities to real identities generated. Here, a centroid tracking algorithm based on Euclidean distances along with the Hungarian algorithm  employed. 

### Open-field (No Entrance)
In any given scenario, the number of mice in a cage is constant. Therefore, any new false positive identities can then be reassigned to its original true positive identity. Similar to the home-cage, we also employed a centroid tracking algorithm based on Euclidean distances along with the Hungarian algorithm

A tutorial of centroid tracking can be found 
[here](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)



#### RFID_Matching 
Customed writted script for track reassocation and RFID assignment



###### RFID Identification and Verification
o


![]()
