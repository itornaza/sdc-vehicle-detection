from classes.dip import dip
from classes.plotting import Plotting
from classes.exploratory_pipeline import *
from classes.data_prep import *
from enum import Enum
from os import sys

class Commands(Enum):
    NONE = 0
    DATA = 1
    IMAGE = 2

def help():
    print()
    print("> -d: Dataset explorations")
    print("> -i: Single image pipeline and plots for inner stages")
    print("> Do not enter flag to run the video pipeline")
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
        print(">>> Data exploration")
        data_prep()
    elif command == Commands.IMAGE:
        print(">>> Single image processing")
        exploratory_pipeline()
        Plotting.exploreColorSpace()
    else:
        print(">>> Create video")
