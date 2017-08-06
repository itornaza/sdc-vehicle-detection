import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dip import dip
from parameters import Prms

# Hog hyper-parameters
COLORSPACE = Prms.COLORSPACE
ORIENT = Prms.ORIENT
PIX_PER_CELL = Prms.PIX_PER_CELL
CELL_PER_BLOCK = Prms.CELL_PER_BLOCK
HOG_CHANNEL = Prms.HOG_CHANNEL

def _get_data_from_file():
    '''
    Reads the images from the ./dataset directory and 
    returns a list with the car images and a list with then non-car images
    '''
    
    # Lists to hold the car and non car images
    car_images = []
    non_car_images = []
    
    # Get the car images from the dataset
    gti_far_path = '../dataset/vehicles/GTI_Far'
    gti_far_images = glob.glob(os.path.join(gti_far_path, '*.png'))
    gti_left_path = '../dataset/vehicles/GTI_Left'
    gti_left_images = glob.glob(os.path.join(gti_left_path, '*.png'))
    gti_middle_path = '../dataset/vehicles/GTI_MiddleClose'
    gti_middle_images = glob.glob(os.path.join(gti_middle_path, '*.png'))
    gti_right_path = '../dataset/vehicles/GTI_Right'
    gti_right_images = glob.glob(os.path.join(gti_right_path, '*.png'))
    kitti_path = '../dataset/vehicles/KITTI_extracted'
    kitti_images = glob.glob(os.path.join(kitti_path, '*.png'))
    
    # Collect results to the cars list
    car_images = gti_far_images + gti_left_images + gti_middle_images + gti_right_images + kitti_images
    
    # Get the non-car images from the dataset
    extras_path = '../dataset/non-vehicles/Extras'
    extras_images = glob.glob(os.path.join(extras_path, '*.png'))
    gti_path = '../dataset/non-vehicles/GTI'
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

def _get_random_images(cars, notcars):
    '''
    Returns a random image for each of the cars and nptcars lists 
    as well as the random index for the car list
    '''
    
    # Just for fun choose random car / not-car indices
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))
    
    # Read in car / not-car images
    car_image = cv2.imread(cars[car_ind])
    notcar_image = cv2.imread(notcars[notcar_ind])

    # Return the images and the random index as well
    return car_image, notcar_image, car_ind

def _hog(car_image):
    '''Applies the hog filter to the image'''

    ### TODO: Convert to whatever color transformation works best
    
    # Convert image to grayscale
    gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    
    # Define HOG parameters from the globals
    orient = ORIENT
    pix_per_cell = PIX_PER_CELL
    cell_per_block = CELL_PER_BLOCK

    # Hog processing
    features, hog_image = dip.get_hog_features(gray,
                                               orient,
                                               pix_per_cell,
                                               cell_per_block,
                                               vis=True,
                                               feature_vec=False)
   # Return the features and the image
    return features, hog_image

def _normalize_features(car_features, notcar_features):
    '''Normalize the features'''
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return X, scaled_X, X_scaler
    
def _processed_dataset(X, scaled_X, car_features_n, notcar_features_n):
    '''Split the dataset into training and test set'''
    
    # Define the labels vector
    y = np.hstack((np.ones(car_features_n), np.zeros(notcar_features_n)))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,
                                                        random_state=rand_state)

    return X_train, X_test, y_train, y_test

#----------------------
# Reporting functions
#----------------------

def _show_stats(data_info):
    print('>>> Dataset with count of', data_info["n_cars"], ' cars and ',
          data_info["n_notcars"], ' non-cars',
          ' of size: ',data_info["image_shape"],
          ' and data type:', data_info["data_type"])

def _show_hog_params():
    print('>>> Using hog with:', ORIENT, 'orientations',
          PIX_PER_CELL, 'pixels per cell and',
          CELL_PER_BLOCK, 'cells per block')

def _show_vector_length(vector_length):
    print('>>> Feature vector length:', vector_length)

#-------------------
# Plot functions
#-------------------

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

def _plot_normalized_features(X, scaled_X, car_image, car_ind):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(car_image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()

def data_prep(vis=True):
    '''Explore the dataset and return the cars and not cars images in two different lists'''
    
    # 1) Get the car and notcar images from the dataset directories
    cars, notcars = _get_data_from_file()
    
    # 2) Get the color and hog features from the random car image
    
    ### TODO:
    car_features, notcar_features = dip.get_combined_features(cars, notcars)
                                        
    # 3) Normalize the combined features
    X, scaled_X, X_scaler = _normalize_features(car_features, notcar_features)
    
    # 4) Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = _processed_dataset(X, scaled_X,
                                                          len(car_features),
                                                          len(notcar_features))

    # Show results by default, if not just return the datasets
    if vis:
        # Get some information about the dataset size
        data_info = _data_look(cars, notcars)
        
        # Get a random car and a random notcar image for display purposes
        car_image, notcar_image, car_ind = _get_random_images(cars, notcars)
        
        # Get the hog features of the random car image
        hog_image_features, hog_image = _hog(car_image)
        
        # Call all ploting functions
        _show_stats(data_info)
        _plot_car_notcar(car_image, notcar_image)
        _show_hog_params()
        _plot_hog(car_image, hog_image)
        _plot_normalized_features(X, scaled_X, car_image, car_ind)
        _show_vector_length(len(X_train[0]))

    # Return the training and testing datasets to be used by the classifier
    return X_train, X_test, y_train, y_test, X_scaler
