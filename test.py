import argparse
from tqdm import tqdm
from glob import glob
import time
import cv2
import os
import numpy as np
import tensorflow as tf

from tools.GuidedFilter import guided_filter
from net.generator import Generator
from net.discriminator import Discriminator


def get_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
            return "GPU"
        except RuntimeError as e:
            print(e)

    print("No GPU found, defaulting to CPU.")
    return "CPU"


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, image_path, hw):
    images = (images.squeeze() + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, (hw[1], hw[0]))
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8:  # resize image to multiple of 8s
        def to_x8s(x):
            return 256 if x < 256 else x - x % 8  # if using tiny model: x - x%16
        img = cv2.resize(img, (to_x8s(w), to_x8s(h)))
    return img/127.5 - 1.0


def load_test_data(image_path, x8=True):
    img0 = cv2.imread(image_path)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocessing(img, x8)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]


def sigm_out_scale(x):
    x = (x + 1.0) / 2.0
    return tf.clip_by_value(x, 0.0, 1.0)


def tanh_out_scale(x):
    x = (x - 0.5) * 2.0
    return tf.clip_by_value(x, -1.0, 1.0)


def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/shinkai',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='inputs/imgs',
                        help='Directory name of test photos')
    parser.add_argument('--save_dir', type=str,
                        default='style_results/', help='Directory name of results')
    return parser.parse_args()


def test(checkpoint_dir, save_dir, test_dir):
    result_dir = check_folder(save_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    # Build models (must match training checkpoint structure)
    gen = Generator(name='generator')
    disc_support = Discriminator(sn=True, ch=32, name='discriminator')
    disc_main = Discriminator(sn=True, ch=32, name='discriminator_main')

    # Initialize with dummy input so variables are created
    dummy = tf.zeros([1, 256, 256, 3])
    gen(dummy, False)
    disc_support(dummy)
    disc_main(dummy)

    # Build optimizers to match checkpoint structure
    G_optim = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
    D_optim = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
    init_G_optim = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)

    # Create checkpoint matching training structure exactly
    checkpoint = tf.train.Checkpoint(
        generator=gen,
        discriminator_support=disc_support,
        discriminator_main=disc_main,
        G_optim=G_optim,
        D_optim=D_optim,
        init_G_optim=init_G_optim,
    )

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        checkpoint.read(latest).expect_partial()
        print(" [*] Success to read {}".format(latest))
    else:
        print(" [*] Failed to find a checkpoint")
        return

    # Load model
    device = get_device()
    print(f"Testing on: {device}")

    imgs = []
    for x in test_files:
        imgs.append(load_test_data(x))

    begin = time.time()
    for i, sample_file in tqdm(list(enumerate(test_files))):
        sample_image, scale = np.asarray(imgs[i][0]), imgs[i][1]

        # Run inference (eager mode)
        sample_input = tf.constant(sample_image, dtype=tf.float32)
        test_s0, test_m = gen(sample_input, False)
        test_s1 = tanh_out_scale(guided_filter(
            sigm_out_scale(test_s0), sigm_out_scale(test_s0), 2, 0.01))

        real = sample_image
        save_images(real, result_dir +
                    '/a_{0}'.format(os.path.basename(sample_file)), scale)
        save_images(test_s1.numpy(), result_dir +
                    '/b_{0}'.format(os.path.basename(sample_file)), scale)
        save_images(test_s0.numpy(), result_dir +
                    '/c_{0}'.format(os.path.basename(sample_file)), scale)
        save_images(test_m.numpy(), result_dir +
                    '/d_{0}'.format(os.path.basename(sample_file)), scale)
    end = time.time()
    print(f'test-time: {end-begin} s')
    print(f'one image test time : {(end-begin)/(len(test_files))} s')


if __name__ == '__main__':
    arg = parse_args()
    print(arg.checkpoint_dir)
    test(arg.checkpoint_dir, arg.save_dir, arg.test_dir)
