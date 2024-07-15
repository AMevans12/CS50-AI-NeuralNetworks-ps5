# CS50 AI Week 5: Neural Networks

## Overview

This repository contains the Traffic project from the neural networks week of the CS50 AI course. The project aims to classify images of traffic signs using convolutional neural networks (CNNs), showcasing the application of deep learning in computer vision tasks.

## Project Description

### Traffic Sign Classification

The Traffic project involves building and training a CNN to recognize and classify different traffic signs accurately. The model is trained on a dataset of traffic sign images and is evaluated based on its ability to correctly identify the signs in new images.

#### Key Features:
- **Data Preprocessing:** Loading and normalizing the dataset.
- **Model Architecture:** Designing and implementing a CNN using layers such as Convolution, Pooling, and Fully Connected layers.
- **Training and Evaluation:** Training the model with the training data and evaluating its performance on the validation set.
- **Prediction:** Using the trained model to classify new images of traffic signs.


## Installation

To use the code in this repository, clone the repository and ensure you have Python and the required libraries installed.


git clone https://github.com/AMevans12/CS50-AI-NeuralNetworks-ps5.git
cd CS50-Neural-Networks-Traffic


Install the required dependencies:

pip install -r requirements.txt


## Usage

### Training the Model

To train the model on the traffic sign dataset:


python traffic.py train


### Evaluating the Model

To evaluate the trained model on the validation set:


python traffic.py evaluate


### Making Predictions

To use the trained model to classify new traffic sign images:


python traffic.py predict path/to/image.jpg


## Contributing

Feel free to fork this repository and submit pull requests for improvements or fixes. Contributions are welcome!

## Contact

For any questions or feedback, please reach out to me via [GitHub](https://github.com/AMevans12).


**Explore neural networks with traffic sign classification!**
