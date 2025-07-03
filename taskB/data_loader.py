# taskB/data_loader.py
import tensorflow as tf
import os
import random
import numpy as np
from itertools import combinations
from .config import IMG_SIZE

def get_image_paths_by_person(directory):
    """Scans a directory and maps each person to their list of image paths."""
    person_images = {}
    if not os.path.exists(directory):
        print(f"Error: Directory not found at {directory}. Please check the path.")
        return None

    person_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    print(f"Scanning directory: {directory}")

    for person_id in person_folders:
        person_dir = os.path.join(directory, person_id)
        image_list = [os.path.join(root, file)
                      for root, _, files in os.walk(person_dir)
                      for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_list) > 1:
            person_images[person_id] = image_list

    print(f"Found {len(person_images)} unique persons with sufficient images.")
    return person_images

def create_image_pairs(person_images, max_samples=None):
    """Generates balanced positive and negative image pairs."""
    anchor_images, comparison_images, labels = [], [], []
    if not person_images:
        return np.array([]), np.array([]), np.array([])

    person_ids = list(person_images.keys())
    print("Generating pairs...")

    limit_per_person = None
    if max_samples and len(person_ids) > 0:
        limit_per_person = max(1, max_samples // (2 * len(person_ids)))
        print(f"Sample limit active. Aiming for ~{limit_per_person} positive pairs per person.")

    for person_id, images in person_images.items():
        positive_pairs = list(combinations(images, 2))
        random.shuffle(positive_pairs)
        if limit_per_person:
            positive_pairs = positive_pairs[:limit_per_person]

        for p1, p2 in positive_pairs:
            anchor_images.append(p1)
            comparison_images.append(p2)
            labels.append(1.0)

        for _ in range(len(positive_pairs)):
            other_person_id = random.choice([pid for pid in person_ids if pid != person_id])
            anchor_img = random.choice(images)
            comparison_img = random.choice(person_images[other_person_id])
            anchor_images.append(anchor_img)
            comparison_images.append(comparison_img)
            labels.append(0.0)

    print(f"Generated {len(anchor_images)} total pairs.")
    if labels:
        print(f"Positive: {int(np.sum(labels))}, Negative: {len(labels) - int(np.sum(labels))}")
    return np.array(anchor_images), np.array(comparison_images), np.array(labels, dtype='float32')

def preprocess_image(image_path):
    """Loads, decodes, resizes, and normalizes an image for MobileNetV2."""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def create_tf_dataset(anchors, comps, labels, batch_size, seed, shuffle=True):
    """Creates a shuffled and batched tf.data.Dataset pipeline."""
    def process_path(anchor_path, comp_path, label):
        return (preprocess_image(anchor_path), preprocess_image(comp_path)), label

    dataset = tf.data.Dataset.from_tensor_slices((anchors, comps, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(anchors), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset