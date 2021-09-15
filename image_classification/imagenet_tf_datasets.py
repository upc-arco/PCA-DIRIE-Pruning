""" Script to prepare the ImageNet2012 dataset after manual download."""

# Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Main
def main():
    print("Preparing ImageNet2012 dataset with Tensorflow Dataset (TFSD)...")

    # Get a dataset builder for the required dataset
    dataset_name = "imagenet2012"
    manual_dataset_dir = "/datasets/imagenet/"
    if dataset_name in tfds.list_builders():
        imagenet_dataset_builder = tfds.builder(dataset_name, data_dir=manual_dataset_dir)
        print("  Retrieved " + dataset_name + " builder")
    else:
        print("  Error getting the builder")
        return

    # Get all the information regarding dataset
    print(imagenet_dataset_builder.info)
    print("  Image Shape: ", imagenet_dataset_builder.info.features['image'].shape)
    print("  Classes: ", imagenet_dataset_builder.info.features['label'].num_classes)
    print("  Labels: ", imagenet_dataset_builder.info.features['label'].names)
    print("  Train Examples: ", imagenet_dataset_builder.info.splits['train'].num_examples)
    print("  Validation Examples: ", imagenet_dataset_builder.info.splits['validation'].num_examples)

    # Prepare the dataset assuming that ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar are in "$manual_dataset_dir/downloads/manual"
    imagenet_dataset_builder.download_and_prepare()

    # End
    print("Finished Preparing ImageNet2012 dataset with Tensorflow Dataset (TFSD).")

if __name__ == "__main__":
    main()