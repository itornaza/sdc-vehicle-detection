## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The project rubric can be found [here](https://review.udacity.com/#!/rubrics/513/view)

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/terminal_training.png
[image4]: ./output_images/test2search.png
[image5]: ./output_images/test1hog.png
[image6]: ./output_images/test2hog.png
[image7]: ./output_images/test3hog.png
[image8]: ./output_images/test4hog.png
[image9]: ./output_images/test5hog.png
[image10]: ./output_images/test6hog.png
[image11]: ./output_images/test1heat.png
[image12]: ./output_images/test2heat.png
[image13]: ./output_images/test3heat.png
[image14]: ./output_images/test4heat.png
[image15]: ./output_images/test5heat.png
[image16]: ./output_images/test6heat.png
[image17]: ./output_images/normalized_features.png

---
### Writeup / README

The project source code can be found in the `./src` directory. To run the main program use the following options: 

* `python main.py -d` builds up the dataset and trains an SVC classifier

* `python main.py -i` runs the vehicle detection pipline on the test images found in `./test_images`. All images in the following analysis are generated with the `-i` option

* `python main.py` runs the vehicle detection pipeline on the `./project_video.mp4` and saves the resulting video with the detectied vehicles in the `./project_video_output.mp4`

Note: The parameters for the hog, heatmap and classifier training are conveniently put in the `parameters.py` file for centralised control.

### Dataset preparation

The dataset is built from the GTI and KITTI car databases found in the `./dataset` directory. All the images contained in this set are in PNG format.

All the code that handles the dataset can be found in the `./src/data_prep.py`. The main handler is the `data_prep()` function on line 209 which in turn calls the internal functions in order to:

* Get the car and notcar images from the dataset directories
* Get the car images features
* Get the not car images features
* Include the flipped images to augment the dataset
* Combine and normalize the features
* Split the dataset into training and test sets

Note that all the functions that are related to digital image processing, transformations and vehicle detection are located in the `dip` class found in the `./src/dip.py` file.

The features normalization can be visualized in the following figure
![alt text][image17]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The car and not car image features for training are extracted by calling the `extract_features()` method of the `dip` class that uses the `combined_features()` and `get_hog_features()` methods of the same class as a helper routines.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of: 
* `hog_channel='ALL'`
* `orientations=9` 
* `pixels_per_cell=(8, 8)`
* `cells_per_block=(2, 2)`

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as `LUV` colorspace with `orientations=8` and `hog_channel=0`, but the black car was very difficult to be detected. Once I set the `ALL` parameter for the `hog_channel` the detection became much more efficient.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `Classifier` class found in the `./src/classifier.py` file. To speed up development I used the `classifier.pkl` file to store the trained classifier and load it from there as I needed it for the next steps.

For the training of the classifier, a 32 x 32 spatial filter and histogram of 32 bins was used in conjuction with the hog features. The respective functions can be found in the `dip` class and the `bin_spatial()` and `color_hist()` methods.

Thw following figure shows an example run of the classifier training:

![alt text][image3]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use the sliding window technique with hog sub sampling to improve the efficiency of the pipeline found in the `find_cars()` method of the `dip` class. I decided to use the following segments of the image for searching as shown in the next figure:

* For the **blue segment** that represents the far field and the cars appear to be smaller. Search is done  from 400 to 500 pixels in the y-axis and from 330 pixels to 1280 pixels for the x-axis with an  overlapping factor of 1.0. The x-axis masking is implemented to avoid the opposing lane cars to be detected for this project.

* The **green segment** that represents the mid range field. Search is done from 400 to 600 pixels in the y-axis and from 160 pixels to 1280 pixels for the x-axis with an  overlapping factor of 1.5. The x-axis masking once again, is implemented to avoid the opposing lane cars to be detected for this project.

* For the **red segment** represents the near field and the cars appear to be bigger. Search is done  from 500 to 650 pixels in the y-axis and from 0 pixels to 1280 pixels for the x-axis with an  overlapping factor of 2.5.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's the youtube [link](https://youtu.be/XEPEyQidjjw) to the video

The pipeline for the video can be found in the `Pipelines` class and the `video_pipeline()` method found in the `./src/pipelines.py` file.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  the `label` function is called at the line 238 of the `video_pipeline()` method of the `Pipelines` class. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Following is an example of the heatmap application on the test images as well as the display of the bounding boxes around the cars:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

To avoid false positives, I used an averaging of 10 frames with a threshold of 28 for the heatmap. This provided a video for the vehicle detection with minimal false positives.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I used the SVC for classifying the vehicles. While this is a fast and efficient method, it has its limitations. Maybe a more sophisticated deep learning technique like YOLO is required for more robust results  

* Hog only worked properly when all channels where used. Whenever I used channel 0 only, the black car was very poorly detected, and the amount of false positives was far more than acceptable.

* The pipeline is crafted for the non-British style driving where the vehicles drive on the right side of the road.

* In order to avoid problems with reading PNG images and having them in the 0-255 range, I created a function in the `dip` class called `read_image()` to import PNG images using OpenCV and convert them to RGB format to be consistent both with the test images and the video. In that way I was able to have the PNGs both as RGB and 0-255. That may seem natural to implement, but it took me a while to handle the image loading consistently.

