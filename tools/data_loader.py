import os
import cv2
import numpy as np
import tensorflow as tf


class ImageGenerator(object):

    def __init__(self, image_dir, image_size, batch_size, num_cpus=8):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.size = image_size
        self.batch_size = batch_size

    def get_image_paths_train(self, image_dir):
        paths = []
        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                continue
            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)
            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue
            paths.append(path_full)
        return paths

    def read_image(self, img_path):
        # Convert from EagerTensor / bytes to Python str
        if hasattr(img_path, 'numpy'):
            img_path = img_path.numpy()
        if isinstance(img_path, bytes):
            img_path = img_path.decode()

        if 'style' in img_path or 'smooth' in img_path:
            # color image1
            image = cv2.imread(img_path)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            image2 = np.zeros(image1.shape).astype(np.float32)
        else:
            # real photo
            image = cv2.imread(img_path)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Color segmentation (ie. region smooth) photo
            image = cv2.imread(img_path.replace(
                'train_photo', "seg_train_5-0.8-50"))
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image1, image2

    def process_image(self, img_path):
        image1, image2 = self.read_image(img_path)
        processing_image1 = image1 / 127.5 - 1.0
        processing_image2 = image2 / 127.5 - 1.0
        return (processing_image1, processing_image2)

    def load_images(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Uniform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # Map path to image  (tf.py_function instead of tf.py_func)
        dataset = dataset.map(
            lambda img_path: tf.py_function(
                self.process_image, [img_path], [tf.float32, tf.float32]),
            num_parallel_calls=self.num_cpus)

        dataset = dataset.batch(self.batch_size)

        return dataset
