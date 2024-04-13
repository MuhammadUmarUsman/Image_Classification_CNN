# Dog vs. Cat Classifier

## Overview

This repository contains a convolutional neural network (CNN) model trained to classify images as either dogs or cats. The model is built using TensorFlow and Keras libraries. It takes images as input and predicts whether the image contains a dog or a cat.

## Dataset
Dataset Used: https://bit.ly/ImgClsKeras.
The dataset consists of two parts:

1. **Training Data**: Input images and corresponding labels (dogs or cats) used to train the model.
2. **Testing Data**: Input images and corresponding labels for evaluating the performance of the trained model.

The images are stored in the 'Dataset' directory, with separate CSV files for input data and labels.

## Model Architecture

The CNN model architecture is as follows:

1. **Input Layer**: Accepts images with dimensions 100x100x3 (RGB).
2. **Convolutional Layers**: Two sets of convolutional layers with ReLU activation followed by max-pooling layers.
3. **Flatten Layer**: Flattens the output from the convolutional layers into a one-dimensional array.
4. **Dense Layers**: Two fully connected dense layers with ReLU activation.
5. **Output Layer**: Dense layer with sigmoid activation, outputting a probability indicating the likelihood of the image being a cat.

## Usage

1. **Training**: Run the provided Python script `train.py` to train the model using the training data.

## Requirements
Ensure you have the following dependencies installed:

Python 3.x
TensorFlow
Keras
NumPy
Pandas
Matplotlib
You can install the required packages using pip:
``` pip install tensorflow keras numpy pandas matplotlib ```
