import time
import os
import tensorflow as tf
from tools.utils import check_folder, str2bool
import argparse
from AnimeGANv3_ghbli import AnimeGANv3


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


"""parsing and configuration"""


def parse_args():
    desc = "AnimeGANv3 — Ghbli Style"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--style_dataset', type=str,
                        default='Ghbli_c1', help='dataset_name')
    parser.add_argument('--dataset', type=str,
                        default='/Users/trognhann/Desktop/AnimeGANv3/dataset', help='dataset directory')
    parser.add_argument('--vgg_dir', type=str,
                        default='/Users/trognhann/Desktop/AnimeGANv3/vgg19_weight', help='vgg19 weight directory')

    parser.add_argument('--init_G_epoch', type=int, default=5,
                        help='The number of epochs for generator initialization')
    parser.add_argument('--epoch', type=int, default=70,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='The size of batch size')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='The number of ckpt_save_freq')
    parser.add_argument('--load_or_resume', type=str.lower, default="load", choices=[
                        "load", "resume"], help='load is used for fine-tuning and resume is used to continue training.')

    parser.add_argument('--init_G_lr', type=float,
                        default=2e-4, help='The generator init learning rate')
    parser.add_argument('--g_lr', type=float, default=5e-5,
                        help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=5e-5,
                        help='Discriminator (support) learning rate')
    parser.add_argument('--d_main_lr', type=float, default=2e-5,
                        help='Discriminator (main) learning rate — lower to weaken D_main')

    # ---------------------------------------------
    parser.add_argument('--img_size', type=int, nargs='+',
                        default=[512, 512], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3,
                        help='The size of image channel')
    parser.add_argument('--sn', type=str2bool, default=True,
                        help='using spectral norm')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""train"""


def train():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    if len(args.img_size) == 1:
        args.img_size = [args.img_size, args.img_size]

    # device info
    device = get_device()
    print(f"Training on: {device}")

    # Create model
    gan = AnimeGANv3(args)

    # Start training
    gan.train()
    print("----- Training finished! -----")


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("start time :", time.strftime(
        "%Y %b %d %H:%M:%S %a", time.localtime(start_time)))
    print("end time :", time.strftime(
        "%Y %b %d %H:%M:%S %a", time.localtime(time.time())))
