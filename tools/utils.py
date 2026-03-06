import cv2
import os
import numpy as np
import tensorflow as tf


def img_resize(img, limit=1280):
    h, w = img.shape[:2]
    max_edge = max(h, w)
    if max_edge > limit:
        scale_factor = limit / max_edge
        height = int(round(h * scale_factor))
        width = int(round(w * scale_factor))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img


def load_test_data(image_path, x8=True):
    img = cv2.imread(image_path)  # .astype(np.float32)
    img = img_resize(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, x8)
    img = np.expand_dims(img, axis=0)
    return img


def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8:  # resize image to multiple of 8s
        def to_8s(x):
            return 256 if x < 256 else x - x % 8  # if using tiny model: x - x%16
        img = cv2.resize(img, (to_8s(w), to_8s(h)))
    return img/127.5 - 1.0


def save_images(images, image_path):
    fake = inverse_transform(images.squeeze())
    return imsave(fake, image_path)


def inverse_transform(images):
    images = (images + 1.) / 2 * 255
    # The calculation of floating-point numbers is inaccurate,
    # and the range of pixel values must be limited to the boundary,
    # otherwise, image distortion or artifacts will appear during display.
    images = np.clip(images, 0, 255)
    return images.astype(np.uint8)


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


def show_all_variables(generator, discriminator_support=None, discriminator_main=None):
    """Print all trainable variables in the given models."""
    print('Generator variables:')
    for var in generator.trainable_variables:
        print(f"  {var.name} {var.shape}")
    if discriminator_support is not None:
        print('Discriminator (support) variables:')
        for var in discriminator_support.trainable_variables:
            print(f"  {var.name} {var.shape}")
    if discriminator_main is not None:
        print('Discriminator (main) variables:')
        for var in discriminator_main.trainable_variables:
            print(f"  {var.name} {var.shape}")


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    """Defines gaussian kernel
    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        2-D Tensor of gaussian kernel
    """
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def gaussian_blur(img, kernel_size=7, sigma=5., ch=3):
    """Convolves a gaussian kernel with input image
    Convolution is performed depthwise
    Args:
        img: 3-D Tensor of image, should by floats
        kernel: 2-D float Tensor for the gaussian kernel
    Returns:
        img: 3-D Tensor image convolved with gaussian kernel
    """
    blur = _gaussian_kernel(kernel_size, sigma, ch, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img
