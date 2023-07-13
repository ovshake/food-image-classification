import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of images using a trained TensorFlow model")
    parser.add_argument('--image_directory', type=str, required=True, help="Directory containing the test images")
    return parser.parse_args()

def build_and_load_model():
    class_names = ["ApplePie", "BagelSandwich", "Bibimbop", "Bread", "FriedRice", "Pork"]
    num_classes = len(class_names)

    base_model = EfficientNetB0(include_top=False, weights='imagenet')

    # Construct the full model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Load the trained model
    checkpoint_filepath = 'checkpoint/trained_model'
    model.load_weights(checkpoint_filepath).expect_partial()

    return model, class_names

def load_and_predict_image(model, image_path, class_names):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict the class of the image
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Write the likelihood of each class
    print(f"Image: {os.path.basename(image_path)}")
    for i, class_name in enumerate(class_names):
        print(f'{class_name}: {score[i]:.2f}', end=" ")
    print("\n")

def main():
    args = parse_args()
    image_directory = args.image_directory

    model, class_names = build_and_load_model()

    for img_file in os.listdir(image_directory):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(image_directory, img_file)
            load_and_predict_image(model, img_path, class_names)

if __name__ == "__main__":
    main()
