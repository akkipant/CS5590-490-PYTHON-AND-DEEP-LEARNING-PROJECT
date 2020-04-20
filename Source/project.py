import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD


# define the show_facial_keypoints function
def show_facial_keypoints(images_input, labels_input, index):
    facial_keypoints = plt.imshow(np.squeeze(images_input[index]), cmap='gray')
    for i in range(15):
        plt.plot(labels_input[index][2 * i], labels_input[index][2 * i + 1], 'ro')
    title = 'The ' + str(index + 1) + ' image with keypoints'
    plt.title(title)
    return facial_keypoints


# define the show projected keypoints function
def show_projected_keypoints(model, images_input, labels_input, index):
    facial_keypoints = plt.imshow(np.squeeze(images_input[index]), cmap='gray')
    predicted = model.predict(images_input[index].reshape(1, 96, 96, 1))
    for i in range(15):
        plt.plot(labels_input[index][2 * i], labels_input[index][2 * i + 1], 'ro')
        plt.plot(predicted[index][2 * i], predicted[index][2 * i + 1], color='blue', marker='o')
    title = 'The ' + str(index + 1) + ' image with actual and projected keypoints'
    plt.title(title)
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


# create a cnn model
def cnn_model(epochs):
    # Create the model
    model = Sequential()
    print('\ninput_shape: ', images_train.shape[1:])

    # Convolutional input layer
    # Dropout layer at 20%.
    model.add(Conv2D(96, (3, 3), input_shape=images_train.shape[1:], padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(30))
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    # model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


# Load data
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('test.csv')
IdLookupTable = pd.read_csv('IdLookupTable.csv')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# remove null values
train_data = null_remove(train_data)

# Explore data
print('The shape of train data : ', train_data.shape)
print('The shape of test data : ', test_data.shape)

print(train_data.head())
print(train_data.columns.values)

# Split the train data into images and labels
images = train_data['Image']
labels = train_data.drop(['Image'], axis=1)
labels = labels.to_numpy()

# explore the train_images and train_keyPoints
number_of_pixels = len(images[0].split(' '))
print('Number of pixels in each image: ', number_of_pixels)
height = width = int(number_of_pixels ** (1 / 2.0))

# Reshape image to height = 96px width = 96px
print(images.shape)
print(images.head())
images = np.vstack(images.apply(lambda x: np.fromstring(x, dtype=float, sep=' ')).values).reshape(
    images.shape[0], height, width, 1)

# scale data
images /= 255.0

# plot the first image
plt.imshow(np.squeeze(images[0]), cmap='gray')
plt.title('The first image with actual and projected keypoints')
plt.show()

# plot the first image with Facial Keypoints
show_facial_keypoints(images, labels, 0)
plt.show()

# split train data and validation data
print(images.shape)
print(labels.shape)
images_train, images_val, labels_train, labels_val = train_test_split(
    images, labels, random_state=42, test_size=.2)
print('The shape of train image: ', images_train.shape)
print('The shape of train label: ', labels_train.shape)
print('The shape of validation image: ', images_val.shape)
print('The shape of validation label: ', labels_val.shape)

show_facial_keypoints(images_val, labels_val, 0)
plt.show()

# # Load saved model
# model = load_model('project.h5')

epochs = 25
model = cnn_model(epochs)
print(model.summary())
# Fit the model
history = model.fit(images_train, labels_train, validation_data=(images_val, labels_val), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(images_val, labels_val, verbose=0)

model.save('project_test.h5')
print("Accuracy: %.2f%%" % (scores[1] * 100))
print('Loss:', scores[0])

# Plot the loss and accuracy using history object
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plot the first image with projected Keypoints of validation data
show_projected_keypoints(model, images_val, labels_val, 0)
plt.show()
