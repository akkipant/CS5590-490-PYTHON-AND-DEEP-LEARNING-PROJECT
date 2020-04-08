import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


# define the show_facial_keypoints function
def show_facial_keypoints(images, keypoints, index):
    facial_keypoints = plt.imshow(images[index], cmap='gray')
    for i in range(15):
        plt.plot(keypoints.loc[index][2 * i], keypoints.loc[index][2 * i + 1], 'ro')
    return facial_keypoints

# check null and remove them
def null_remove(train_data):
    # check null values
    print('\nCount the null value in the train data: ')
    print(train_data.isnull().any().describe())

    # drop the na value
    train_data = train_data.dropna()
    print('\nAfter dropping the null value, Count the null value in the train data: ')
    print(train_data.isnull().any().describe())

    print('\nCount the null value in the test data: ')
    print(test_data.isnull().any().describe())
    print('\nCount the null value in the ID Lookup Table: ')
    print(IdLookupTable.isnull().any().describe())
    return train_data


# Load data
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('test.csv')
IdLookupTable = pd.read_csv('IdLookupTable.csv')

# remove null values
null_remove(train_data)

# Explore data
print('The shape of train data : ', train_data.shape)
print('The shape of test data : ', test_data.shape)

print(train_data.head())
print(train_data.columns.values)

# Split the train data into train_images and train_keyPoints
train_images = train_data['Image']
train_labels = train_data.drop(['Image'], axis=1)

# explore the train_images and train_keyPoints
number_of_pixels = len(train_images[0].split(' '))
print('Number of pixels in each image: ', number_of_pixels)
height = width = int(number_of_pixels ** (1 / 2.0))

# Reshape image to height = 96px width = 96px
print(train_images.shape)
print(train_images.head())
train_images = train_images.apply(lambda pixel: np.fromstring(pixel, sep=' ').reshape((height, width)))

# scale data
train_images /= 255.0
print(train_images.head())

# plot the first image
plt.imshow(train_images[0], cmap='gray')
plt.show()


# plot the first image with Facial Keypoints
show_facial_keypoints(train_images, train_labels, 0)
plt.show()