import os
import nnfs
from cv2 import cv2
from nnfs.datasets import spiral_data, sine_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AccuracyCategorical import AccuracyCategorical
from AccuracyRegression import AccuracyRegression
from ActivationLinear import ActivationLinear
from ActivationSigmoid import ActivationSigmoid
from ActivationSoftmaxLossCategoticalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU
from LossBinaryCrossentropy import LossBinaryCrossentropy
from LossMeanSquaredError import LossMeanSquaredError
from Model import Model
from OptimizerAdam import OptimizerAdam
from LayerDropout import LayerDropout
from OptimizerSGD import OptimizerSGD
from ActivationSoftmax import ActivationSoftmax
from LossCategoricalCrossentropy import LossCategoricalCrossentropy


# Loads a MNIST dataset
def load_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_dataset('train', path)
    X_test, y_test = load_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], - 1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], - 1).astype(np.float32) -
          127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist.model')

# Evaluate the model
model.evaluate(X_test, y_test)

