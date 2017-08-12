
from classifier import My_classifier
from data_prep import *
from dip import dip
from pipelines import Pipelines
from parameters import Prms
from plotting import Plotting

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from moviepy.editor import VideoFileClip
from os import sys
from scipy.ndimage.measurements import label

video_in_test = '../test_video.mp4'
video_in = '../project_video.mp4'
video_out = '../project_video_output.mp4'

class Commands(Enum):
    NONE = 0
    DATA = 1
    IMAGE = 2

def help():
    print()
    print("> -d: Dataset set up and classifier training")
    print("> -i: Test the classifier on images")
    print("> Run the classifier on video")
    print()

def parseCommands():
    '''Parse the command line arguments'''
    
    command = Commands.NONE
    if len(sys.argv) > 1:
        if sys.argv[1] == '-d':
            command = Commands.DATA
        elif sys.argv[1] == '-i':
            command = Commands.IMAGE
        else:
            help()
            exit(0)

    return command

if __name__ == '__main__':
    command = parseCommands()
    if command == Commands.DATA:
        print(">>> Setting up dataset and training the classifier")

        # 1) Explore the colorspace
        Plotting.exploreColorSpace()
        
        # 2) Get the training and test datasets
        X_train, X_test, y_train, y_test, X_scaler = data_prep(vis=True)
        save_scaler(X_scaler)
        
        # 3) Train the classifier
        svc = My_classifier.classify(X_train, X_test, y_train, y_test, vis=True)
        My_classifier.save(svc)
    
    elif command == Commands.IMAGE:
        print(">>> Testing the classifier on images")
        
        # 1) Load the classifier and the scaler
        svc = My_classifier.load()
        X_scaler = load_scaler()
        
        # TODO: Uncomment after video ok
        # 2) Test the classifier on test images
        #print(">>> Displaying sliding window processed images")
        #Pipelines.hot_windows(svc, X_scaler, vis=True)

        # TODO: Uncomment after video ok
        # 3) Test classifier and hog sub sampling
        #print(">>> Displaying sub sampling procesed images")
        #Pipelines.hog_sub_sampling(svc, X_scaler)

        # 4) Use the heatmap
        print(">>> Displaing the heatmap of the sub sampling prpcessed images")
        Pipelines.heat(svc, X_scaler)

    else:
        print(">>> Running the classifier on video")
        video = video_in_test if Prms.DEBUG else video_in
        clip = VideoFileClip(video)
        white_clip = clip.fl_image(Pipelines.video_pipeline)
        white_clip.write_videofile(video_out, audio=False)
