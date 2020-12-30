## Imports ##

import sys
import os
from multiprocessing import Process

## Args ##
prefix = '../Video'
def getPrefixDir(index):
    global prefix
    return os.path.join(prefix, str(index))

def getFrontCameraDir(index):
    return os.path.join(getPrefixDir(index), 'front')

def getFrontCameraFilename(index):
    return os.path.join(getFrontCameraDir(index), 'front_camera.mp4')
    
def getBackCameraDir(index):
    return os.path.join(getPrefixDir(index), 'back')

def getBackCameraFilename(index):
    return os.path.join(getBackCameraDir(index), 'back_camera.mp4')

def getHighlightFilename(index):
    return os.path.join(getHighlightDir(index), getBackCameraFilename(index).split('/')[-1])

def getHighlightDir(index):
    return os.path.join(getPrefixDir(index), 'highlight')
    
def getMinDuration():
    return '-1'
    
def getMaxDuration():
    return '-1'

## Core Functions ##

def genHighlights(index):
    # arg1 : filepath to back video
    # arg2 : filepath to front video
    # arg3 : filepath to directory to put highlight
    # arg4 : int representing min highlight duration
    # arg5 : int representing max highlight duration
    args = [getBackCameraFilename(index), getFrontCameraFilename(index), getHighlightDir(index), 
            getMinDuration(), getMaxDuration()]
    command = "python3 main.py " + ' '.join(args)
    print("Executing %s" % command)
    os.system(command)

def genAll():
    global prefix
    for file in os.listdir(prefix):
        if (os.path.isdir(os.path.join(prefix, file))):
            try:
                index = str(int(file))
                videos = [file for file in os.listdir(getHighlightDir(index)) if file.endswith('.mp4')]
                # No highlight already generated
                if (len(videos) == 0):
                    print('Generating higlight for index %s' % index)
                    genHighlights(index)
            except Exception as e:
                print(e)
                continue
       
## Main ##

if __name__ == '__main__':
    genAll()
