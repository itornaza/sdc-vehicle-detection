import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os

def data_prep():
    
    # Vehicles
    gti_far_path = './dataset/vehicles/GTI_Far'
    gti_far_images = glob.glob(os.path.join(gti_far_path, '*.png'))
    print(len(gti_far_images))
    
    gti_left_path = './dataset/vehicles/GTI_Left'
    gti_left_images = glob.glob(os.path.join(gti_left_path, '*.png'))
    print(len(gti_left_images))
    
    gti_middle_path = './dataset/vehicles/GTI_MiddleClose'
    gti_middle_images = glob.glob(os.path.join(gti_middle_path, '*.png'))
    print(len(gti_middle_images))
    
    gti_right_path = './dataset/vehicles/GTI_Right'
    gti_right_images = glob.glob(os.path.join(gti_right_path, '*.png'))
    print(len(gti_right_images))
    
    kitti_path = './dataset/vehicles/KITTI_extracted'
    kitti_images = glob.glob(os.path.join(kitti_path, '*.png'))
    print(len(kitti_images))
    
    # Non-vehicles
    extras_path = './dataset/non-vehicles/Extras'
    extras_images = glob.glob(os.path.join(extras_path, '*.png'))
    print(len(extras_images))
    
    gti_path = './dataset/non-vehicles/GTI'
    gti_images = glob.glob(os.path.join(gti_path, '*.png'))
    print(len(gti_images))
    
    '''
    cars = []
    notcars = []

    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])


    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    '''
