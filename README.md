# Food Image Classification

This project is about classifying different food images using a Convolutional Neural Network (CNN) with the EfficientNetB0 model as the base model, implemented in TensorFlow.

## Requirements

- Python 3.8+
- TensorFlow 2.4+
- Glob

## Usage

First, install the necessary libraries using pip:

```
pip install -r requirements.txt
```


After installing the necessary libraries, you can run the script by navigating to the project directory and entering the following command in your terminal:

```
python script_name.py --image_directory /path/to/your/image/directory
```


Replace `/path/to/your/image/directory` with the path to your directory of images. The script expects images to be in subdirectories under the main directory, where each subdirectory name is the class name.

For example, if you're classifying images of "ApplePie", "BagelSandwich", etc., your directory structure might look something like this:

```
image_directory
|
└───ApplePie
|   |   image1.jpg
|   |   image2.jpg
|   |   ...
└───BagelSandwich
|   |   image1.jpg
|   |   image2.jpg
|   |   ...
└───...
```


## Training

The script will split the dataset into training and validation sets (80% for training, 20% for validation), compile the model using Adam optimizer, and then start training the model for 30 epochs.

The best weights, according to the validation accuracy, will be saved in a directory named `checkpoint` under the filename `trained_model`.

## Inference
Use the script `inference.py` as follows
```
python inference.py --image_directory /path/to/folder/containining/images
```
This will automatically pick up the best trained checkpoint and run it on the images in your directory. Please ensure the images have the `.jpg` extension. This will give outputs like this
```
1/1 [==============================] - 3s 3s/step
Image: A020511XX_02577.jpg
ApplePie: 0.35 BagelSandwich: 0.13 Bibimbop: 0.13 Bread: 0.13 FriedRice: 0.13 Pork: 0.13
```

## Current State

As of now, this script is set to classify six types of food: "ApplePie", "BagelSandwich", "Bibimbop", "Bread", "FriedRice", and "Pork". If you want to classify different types of food, you need to change the `class_names` list in the script accordingly and ensure you have the corresponding subdirectories with images in your image directory.
