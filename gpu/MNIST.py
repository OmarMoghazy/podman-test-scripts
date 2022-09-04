# Import Tensorflow and check version
import tensorflow as tf
# Import TensorFlow Keras
from tensorflow import keras
#Import numpy
import numpy as np

tf.get_logger().setLevel('ERROR')

# Import the dataset.

fashion_mnist = keras.datasets.fashion_mnist

# Load data set as four numpy arrays: 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# For training the model, we will use train_images and train_labels arrays.
# To test the performance of the trained model, we are going to use the test_images and test_labels arrays.

print(f'There are {len(train_images)} images in the training set and {len(test_images)} images in the testing set.')

print(f'There are {len(train_labels)} labels in the training set and {len(test_labels)} labels in the test set.')

print(f'The images are {train_images[0][0].size} x {train_images[0][1].size} NumPy arrays.')

label_cloth_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 
                    3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 
                    7:'Sneaker', 8:'Bag', 9:'Ankle boot' }
def label_name(x):
    return label_cloth_dict[x]

# The pixel values range from 0 to 255. 
# Let's divide the image arrays by 255 to scale them to the range 0 to 1.

train_images = train_images / 255.0

test_images = test_images / 255.0
# Let's build the model:

simple_model = keras.Sequential([
    # Flatten two dimansional images into one dimansion 28*28pixles=784pixels.
    keras.layers.Flatten(input_shape=(28, 28)),
    # First dense/ fully connected layer: 128 nodes.
    keras.layers.Dense(128, activation='relu'),
    # Second dense/ fully connected layer: 10 nodes --> Result is a score for each images class.
    keras.layers.Dense(10)])

# Compile the model:
# Define loss function, optimizer, and metrics.

simple_model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'])

# Train the model:
# Let's train 15 epochs. After every epoch, training time, loss, and accuracy will be displayed.

simple_model.fit(train_images, train_labels, epochs=15)

# Let's see how the model performs on the test data:

test_loss, test_acc = simple_model.evaluate(test_images, test_labels)
