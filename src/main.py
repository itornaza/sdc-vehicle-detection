from classifier import My_classifier
from dip import dip
from plotting import Plotting
from pipelines import Pipelines
from data_prep import *
from enum import Enum
from os import sys

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
        
        # 2) Test the classifier on test images
        Pipelines.hot_windows(svc, X_scaler, vis=True)

        # 3) Test classifier and hog sub sampling
        Pipelines.hog_sub_sampling

    else:
        print(">>> Running the classifier on video")
