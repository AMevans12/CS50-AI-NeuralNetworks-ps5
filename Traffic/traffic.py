import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
image_size = (IMG_WIDTH, IMG_HEIGHT)

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=123
    )

    # Normalize images to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a numpy array of
    all images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a numpy array of integer labels, representing the categories for each
    of the corresponding images.
    """
    images = []
    labels = []

    # Loop over each category directory
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        
        # Loop over each image in the category directory
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))  # Use 'softmax' for multi-class classification
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    main()
