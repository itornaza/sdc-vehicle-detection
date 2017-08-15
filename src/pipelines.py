
from classifier import My_classifier
from data_prep import *
from dip import dip
from parameters import Prms

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

class Pipelines:

    # Parameters for frame processing
    frame_n = 0 # Keeps track of the frames
    frame_group_box_list = [] # Box list for a group of frames
    last_full_box_list = [] # Last full box list for a group of frames
    
    # Note: The max numner of frames is set in the Parameters class

    def hot_windows(svc, X_scaler, vis=False):
        '''Check the classifier by applying the vehicle detection to the test images'''

        for img in glob.glob('../test_images/test*.jpg'):
            image = dip.read_image(img)
            draw_image = np.copy(image)

            windows = dip.slide_window(image,
                                       x_start_stop=[None, None],
                                       y_start_stop=[Prms.Y_START, Prms.Y_STOP],
                                       xy_window=Prms.XY_WINDOW,
                                       xy_overlap=Prms.XY_OVERLAP)
                        
            # Keep the windows in each image
            hot_windows = dip.search_windows(image,
                                             windows,
                                             svc,
                                             X_scaler,
                                             color_space=Prms.COLORSPACE,
                                             spatial_size=Prms.SPATIAL_SIZE,
                                             hist_bins=Prms.N_BINS,
                                             orient=Prms.ORIENT,
                                             pix_per_cell=Prms.PIX_PER_CELL,
                                             cell_per_block=Prms.CELL_PER_BLOCK,
                                             hog_channel=Prms.HOG_CHANNEL,
                                             spatial_feat=Prms.SPATIAL_FEAT,
                                             hist_feat=Prms.HIST_FEAT,
                                             hog_feat=Prms.HOG_FEAT)
                    
            # Plot the vehicle detection image
            window_img = dip.draw_boxes(draw_image, hot_windows, color=Prms.LINE_COLOR, thick=Prms.LINE_THICKNESS)
            plt.imshow(window_img)
            plt.show()

    def hog_sub_sampling(svc, X_scaler):
        '''Apply hog sub-sampling to an image to locate cars with one search'''

        for img in glob.glob('../test_images/test*.jpg'):
            image = dip.read_image(img)
            
            # Just run for the middle distance detection
            out_img, box_list = dip.find_cars(image,
                                              Prms.Y_START[Prms.MID],
                                              Prms.Y_STOP[Prms.MID],
                                              Prms.SCALE[Prms.MID],
                                              svc, X_scaler,
                                              Prms.HOG_CHANNEL,
                                              Prms.ORIENT,
                                              Prms.PIX_PER_CELL,
                                              Prms.CELL_PER_BLOCK,
                                              Prms.SPATIAL_SIZE,
                                              Prms.N_BINS)
            
            # Display the results
            plt.imshow(out_img)
            plt.show()

    def heat(svc, X_scaler):
        '''Apply a heat map on the test images to validate performance'''

        for img in glob.glob('../test_images/test*.jpg'):
            image = dip.read_image(img)
            
            # Create an empty heat map to draw on
            heat = np.zeros_like(image[:,:,0]).astype(np.float)

            # Get the box list from using the hog sub sampling technique
            out_img, box_list_far = dip.find_cars(image,
                                                  Prms.Y_START[Prms.FAR],
                                                  Prms.Y_STOP[Prms.FAR],
                                                  Prms.SCALE[Prms.FAR],
                                                  svc, X_scaler,
                                                  Prms.HOG_CHANNEL,
                                                  Prms.ORIENT,
                                                  Prms.PIX_PER_CELL,
                                                  Prms.CELL_PER_BLOCK,
                                                  Prms.SPATIAL_SIZE,
                                                  Prms.N_BINS)
            
            out_img, box_list_mid = dip.find_cars(image,
                                                  Prms.Y_START[Prms.MID],
                                                  Prms.Y_STOP[Prms.MID],
                                                  Prms.SCALE[Prms.MID],
                                                  svc, X_scaler,
                                                  Prms.HOG_CHANNEL,
                                                  Prms.ORIENT,
                                                  Prms.PIX_PER_CELL,
                                                  Prms.CELL_PER_BLOCK,
                                                  Prms.SPATIAL_SIZE,
                                                  Prms.N_BINS)
            
            out_img, box_list_near = dip.find_cars(image,
                                                   Prms.Y_START[Prms.NEAR],
                                                   Prms.Y_STOP[Prms.NEAR],
                                                   Prms.SCALE[Prms.NEAR],
                                                   svc, X_scaler,
                                                   Prms.HOG_CHANNEL,
                                                   Prms.ORIENT,
                                                   Prms.PIX_PER_CELL,
                                                   Prms.CELL_PER_BLOCK,
                                                   Prms.SPATIAL_SIZE,
                                                   Prms.N_BINS)

            # out_images are discarded at this time
            box_list = box_list_far + box_list_mid + box_list_near

            # Add heat to each box in box list
            heat = dip.add_heat(heat, box_list)

            # Apply threshold to help remove false positives
            heat = dip.apply_threshold(heat, Prms.IMAGE_THRESHOLD)

            # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img = dip.draw_labeled_bboxes(np.copy(image), labels)

            # Display the results
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()

    def video_pipeline(image):
        '''
        The main pipeline to process the video from the front camera of the car and
        returns the video with the detected vehicles
        '''
        
        # Frames book-keeping
        if Pipelines.frame_n >= Prms.FRAMES_MAX:
            # Reset the number of frames counter
            Pipelines.frame_n = 0
            
            # We have processed the max number of frames so we can store
            # the sum of their box lists to. The assignment is passed by value
            # rather than reference
            Pipelines.last_full_box_list = Pipelines.frame_group_box_list[:]
            
            # Reset the frame group box list to process a new group of frames
            Pipelines.frame_group_box_list[:] = []
        
        # Increase the frame for the next itteration
        Pipelines.frame_n = Pipelines.frame_n + 1
        
        # Load the classifier and the scaler
        svc = My_classifier.load()
        X_scaler = load_scaler()

        # Create an empty heat map to draw on
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        
        # Get the box list from using the hog sub sampling technique
        # For the far field
        out_img, box_list_far = dip.find_cars(image,
                                              Prms.Y_START[Prms.FAR],
                                              Prms.Y_STOP[Prms.FAR],
                                              Prms.SCALE[Prms.FAR],
                                              svc, X_scaler,
                                              Prms.HOG_CHANNEL,
                                              Prms.ORIENT,
                                              Prms.PIX_PER_CELL,
                                              Prms.CELL_PER_BLOCK,
                                              Prms.SPATIAL_SIZE,
                                              Prms.N_BINS,
                                              Prms.X_START[Prms.FAR])

        # Mid field
        out_img, box_list_mid = dip.find_cars(image,
                                              Prms.Y_START[Prms.MID],
                                              Prms.Y_STOP[Prms.MID],
                                              Prms.SCALE[Prms.MID],
                                              svc, X_scaler,
                                              Prms.HOG_CHANNEL,
                                              Prms.ORIENT,
                                              Prms.PIX_PER_CELL,
                                              Prms.CELL_PER_BLOCK,
                                              Prms.SPATIAL_SIZE,
                                              Prms.N_BINS,
                                              Prms.X_START[Prms.MID])
        
        # Near field
        out_img, box_list_near = dip.find_cars(image,
                                               Prms.Y_START[Prms.NEAR],
                                               Prms.Y_STOP[Prms.NEAR],
                                               Prms.SCALE[Prms.NEAR],
                                               svc, X_scaler,
                                               Prms.HOG_CHANNEL,
                                               Prms.ORIENT,
                                               Prms.PIX_PER_CELL,
                                               Prms.CELL_PER_BLOCK,
                                               Prms.SPATIAL_SIZE,
                                               Prms.N_BINS,
                                               Prms.X_START[Prms.NEAR])
        
        # Append the local and global box list
        box_list = box_list_far + box_list_mid + box_list_near
        Pipelines.frame_group_box_list += box_list

        # Add heat to each box in box list
        heat = dip.add_heat(heat, Pipelines.last_full_box_list)

        # Apply threshold to help remove false positives
        heat = dip.apply_threshold(heat, Prms.VIDEO_THRESHOLD)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = dip.draw_labeled_bboxes(np.copy(image), labels)

        # Return the image with the detected vehicles
        return draw_img
