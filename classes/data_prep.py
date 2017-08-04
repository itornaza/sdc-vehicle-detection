import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

from classes.dip import dip

def _get_data_from_file():
    '''
    Reads the images from the ./dataset directory and 
    returns a list with the car images and a list with then non-car images
    '''
    
    # Lists to hold the car and non car images
    car_images = []
    non_car_images = []
    
    # Get the car images from the dataset
    gti_far_path = './dataset/vehicles/GTI_Far'
    gti_far_images = glob.glob(os.path.join(gti_far_path, '*.png'))
    gti_left_path = './dataset/vehicles/GTI_Left'
    gti_left_images = glob.glob(os.path.join(gti_left_path, '*.png'))
    gti_middle_path = './dataset/vehicles/GTI_MiddleClose'
    gti_middle_images = glob.glob(os.path.join(gti_middle_path, '*.png'))
    gti_right_path = './dataset/vehicles/GTI_Right'
    gti_right_images = glob.glob(os.path.join(gti_right_path, '*.png'))
    kitti_path = './dataset/vehicles/KITTI_extracted'
    kitti_images = glob.glob(os.path.join(kitti_path, '*.png'))
    
    # Collect results to the cars list
    car_images = gti_far_images + gti_left_images + gti_middle_images + gti_right_images + kitti_images
    
    # Get the non-car images from the dataset
    extras_path = './dataset/non-vehicles/Extras'
    extras_images = glob.glob(os.path.join(extras_path, '*.png'))
    gti_path = './dataset/non-vehicles/GTI'
    gti_images = glob.glob(os.path.join(gti_path, '*.png'))
    
    # Collect results to the non-cars list
    non_car_images = extras_images + gti_images

    # Return a list for the car and a list for the non-car images
    return car_images, non_car_images

def _data_look(car_list, notcar_list):
    '''Returns some characteristics of the dataset'''
        
    data_dict = {}
    
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    
    # Read in a test image, either car or notcar
    example_img = cv2.imread(car_list[0])
    
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    
    # Return data_dict
    return data_dict

def _show_stats(data_info):
    print('Your function returned a count of', data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', data_info["data_type"])

def _get_random_images(cars, notcars):
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))
    
    # Read in car / not-car images
    car_image = cv2.imread(cars[car_ind])
    notcar_image = cv2.imread(notcars[notcar_ind])

    return car_image, notcar_image

def _hog(car_image):
    '''Applies the hog filter to the image'''

    # Convert image to grayscale
    gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    # Hog processing
    features, hog_image = dip.get_hog_features(gray, orient,
                                               pix_per_cell,
                                               cell_per_block,
                                               vis=True,
                                               feature_vec=False)

    return features, hog_image

def _plot_car_notcar(car_image, notcar_image):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.show()

def _plot_hog(car_image, hog_image):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()

def data_prep(vis=True):
    '''Explore the dataset and return the cars and not cars images in two different lists'''
    
    cars, notcars = _get_data_from_file()
    data_info = _data_look(cars, notcars)
    car_image, notcar_image = _get_random_images(cars, notcars)
    features, hog_image = _hog(car_image)
    
    # TODO: Combine and normalize features
    
    # Show results by default
    if vis:
        _show_stats(data_info)
        _plot_car_notcar(car_image, notcar_image)
        _plot_hog(car_image, hog_image)

    return cars, notcars
