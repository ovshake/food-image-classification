import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using TensorFlow and EfficientNetB0")
    parser.add_argument('--image_directory', type=str, required=True, help="Directory containing the images")
    return parser.parse_args()

def count_images(image_directory):
    return len(glob(image_directory + '/*/*.jpg'))

def get_dataset(image_directory, subset_type, class_names):
    return image_dataset_from_directory(
        image_directory,
        class_names=class_names,
        label_mode='categorical',
        validation_split=0.2,
        subset=subset_type,
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )

def get_model_checkpoint_callback():
    return ModelCheckpoint(
        filepath='checkpoint/trained_model',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

def get_model(num_classes):
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_ds, val_ds):
    model_checkpoint_callback = get_model_checkpoint_callback()
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[model_checkpoint_callback])

def main():
    args = parse_args()
    image_directory = args.image_directory

    total_images = count_images(image_directory)
    print(f"Total images: {total_images}")

    class_names = ["ApplePie", "BagelSandwich", "Bibimbop", "Bread", "FriedRice", "Pork"]
    train_ds = get_dataset(image_directory, 'training', class_names)
    val_ds = get_dataset(image_directory, 'validation', class_names)

    print(f"Class Names: {train_ds.class_names}")

    model = get_model(len(class_names))
    train_model(model, train_ds, val_ds)

if __name__ == "__main__":
    main()
