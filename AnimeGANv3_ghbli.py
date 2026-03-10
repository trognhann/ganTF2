import tensorflow as tf
from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
import cv2
from joblib import Parallel, delayed
from skimage import segmentation, color
from net.generator import Generator
from net.discriminator import Discriminator
from tools.data_loader import ImageGenerator
from tools.GuidedFilter import guided_filter
from tools.L0_smoothing import L0Smoothing

class AnimeGANv3(object):
    def __init__(self, args):
        self.model_name = 'AnimeGANv3'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.style_dataset
        self.dataset = args.dataset
        init_vgg(args.vgg_dir)

        self.epoch = args.epoch
        self.init_G_epoch = args.init_G_epoch

        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        self.load_or_resume = args.load_or_resume

        self.init_G_lr = args.init_G_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # Build models
        self.gen = Generator(name='generator')
        self.disc_support = Discriminator(
            sn=self.sn, ch=32, name='discriminator')
        self.disc_main = Discriminator(
            sn=self.sn, ch=32, name='discriminator_main')

        # Data generators
        self.real_generator = ImageGenerator(
            os.path.join(self.dataset, 'train_photo'), self.img_size, self.batch_size)
        self.anime_image_generator = ImageGenerator(
            os.path.join(self.dataset, self.dataset_name, 'style'), self.img_size, self.batch_size)
        self.anime_smooth_generator = ImageGenerator(
            os.path.join(self.dataset, self.dataset_name, 'smooth'), self.img_size, self.batch_size)
        self.dataset_num = max(
            self.real_generator.num_images, self.anime_image_generator.num_images)

        # Optimizers
        self.init_G_optim = tf.keras.optimizers.Adam(
            self.init_G_lr, beta_1=0.5, beta_2=0.999)
        self.G_optim = tf.keras.optimizers.Adam(
            self.g_lr, beta_1=0.5, beta_2=0.999)
        self.D_optim = tf.keras.optimizers.Adam(
            self.d_lr, beta_1=0.5, beta_2=0.999)

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(
            generator=self.gen,
            discriminator_support=self.disc_support,
            discriminator_main=self.disc_main,
            G_optim=self.G_optim,
            D_optim=self.D_optim,
            init_G_optim=self.init_G_optim,
        )

        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_G_epoch : ", self.init_G_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# init_G_lr,g_lr,d_lr : ", self.init_G_lr, self.g_lr, self.d_lr)
        print()

    def sigm_out_scale(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        x = (x + 1.0) / 2.0
        return tf.clip_by_value(x, 0.0, 1.0)

    def tanh_out_scale(self, x):
        """
        @param x: image tensor  [0.0, 1.0]
        @return:  image tensor  [-1.0, 1.0]
        """
        x = (x - 0.5) * 2.0
        return tf.clip_by_value(x, -1.0, 1.0)

    def compute_generator_output(self, real_photo, is_training):
        """Forward pass through generator + guided filter."""
        generated_s, generated_m = self.gen(real_photo, is_training)
        generated = self.tanh_out_scale(guided_filter(
            self.sigm_out_scale(generated_s),
            self.sigm_out_scale(generated_s), 2, 0.01))
        return generated_s, generated_m, generated

    @tf.function
    def pretrain_G_step(self, real_photo):
        """Pre-training step for generator."""
        with tf.GradientTape() as tape:
            generated_s, generated_m, generated = self.compute_generator_output(
                real_photo, True)
            pre_loss = con_loss(real_photo, generated) + \
                con_loss(real_photo, generated_m)

        G_vars = self.gen.trainable_variables
        grads = tape.gradient(pre_loss, G_vars)
        grads_and_vars = [(g, v)
                          for g, v in zip(grads, G_vars) if g is not None]
        self.init_G_optim.apply_gradients(grads_and_vars)
        return pre_loss

    @tf.function
    def train_G_step(self, real_photo, photo_superpixel, anime, anime_smooth,
                     fake_superpixel, fake_NLMean_l0):
        """Generator training step."""
        with tf.GradientTape() as tape:
            generated_s, generated_m, generated = self.compute_generator_output(
                real_photo, True)

            # gray mapping (Gray-scale Style Loss requirement to maintain pastel tones)
            fake_sty_gray = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(generated))
            anime_sty_gray = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(anime))

            # discriminator predictions
            fake_gray_logit = self.disc_support(fake_sty_gray)
            generated_m_logit = self.disc_main(generated_m)

            # Losses for Support Tail
            _con_loss = con_loss(real_photo, generated, 0.5)
            # Weights exactly aligned with DTGAN manuscript for Gray-scale style loss mapping
            s22, s33, s44 = style_loss_decentralization_3(
                anime_sty_gray, fake_sty_gray, [0.1, 5.0, 25.0])
            _sty_loss = s22 + s33 + s44
            _rs_loss = region_smoothing_loss(fake_superpixel, generated, 0.2) \
                + VGG_LOSS(photo_superpixel, generated) * 0.2
            # 20.0 strictly follows lambda_2 in the DTGAN manuscript
            _color_loss = Lab_color_loss(real_photo, generated, 20.0)
            _tv_loss = 0.001 * total_variation_loss(generated)
            _g_adv_loss = generator_loss(fake_gray_logit)

            G_support_loss = _g_adv_loss + _con_loss + \
                _sty_loss + _rs_loss + _color_loss + _tv_loss

            # Losses for Main Tail
            _tv_loss_m = 0.001 * total_variation_loss(generated_m)
            _p4_loss = VGG_LOSS(fake_NLMean_l0, generated_m) * 0.5
            _p0_loss = L1_loss(fake_NLMean_l0, generated_m) * 50.0
            _g_m_loss = generator_loss_m(generated_m_logit) * 0.02

            G_main_loss = _g_m_loss + _p0_loss + _p4_loss + _tv_loss_m
            Generator_loss = G_support_loss + G_main_loss

        G_vars = self.gen.trainable_variables
        grads = tape.gradient(Generator_loss, G_vars)
        grads_and_vars = [(g, v)
                          for g, v in zip(grads, G_vars) if g is not None]
        self.G_optim.apply_gradients(grads_and_vars)

        return {
            'G_loss': Generator_loss,
            'G_support_loss': G_support_loss,
            'g_adv_loss': _g_adv_loss,
            'con_loss': _con_loss,
            'rs_loss': _rs_loss,
            'sty_loss': _sty_loss,
            's22': s22, 's33': s33, 's44': s44,
            'color_loss': _color_loss,
            'tv_loss': _tv_loss,
            'G_main_loss': G_main_loss,
            'g_m_loss': _g_m_loss,
            'p0_loss': _p0_loss,
            'p4_loss': _p4_loss,
            'tv_loss_m': _tv_loss_m,
        }

    @tf.function
    def train_D_step(self, real_photo, photo_superpixel, anime, anime_smooth,
                     fake_superpixel, fake_NLMean_l0):
        """Discriminator training step."""
        with tf.GradientTape() as tape:
            generated_s, generated_m, generated = self.compute_generator_output(
                real_photo, True)

            # gray mapping
            fake_sty_gray = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(generated))
            anime_sty_gray = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(anime))
            gray_anime_smooth = tf.image.grayscale_to_rgb(
                tf.image.rgb_to_grayscale(anime_smooth))

            # discriminator predictions (support)
            gray_anime_smooth_logit = self.disc_support(gray_anime_smooth)
            anime_gray_logit = self.disc_support(anime_sty_gray)
            fake_gray_logit = self.disc_support(fake_sty_gray)

            # discriminator predictions (main)
            generated_m_logit = self.disc_main(generated_m)
            fake_NLMean_logit = self.disc_main(fake_NLMean_l0)

            D_support_loss = discriminator_loss(anime_gray_logit, fake_gray_logit) \
                + discriminator_loss_346(gray_anime_smooth_logit) * 2.0
            D_main_loss = discriminator_loss_m(
                fake_NLMean_logit, generated_m_logit) * 0.1
            Discriminator_loss = D_support_loss + D_main_loss

        D_vars = self.disc_support.trainable_variables + \
            self.disc_main.trainable_variables
        grads = tape.gradient(Discriminator_loss, D_vars)
        grads_and_vars = [(g, v)
                          for g, v in zip(grads, D_vars) if g is not None]
        self.D_optim.apply_gradients(grads_and_vars)

        return {
            'D_loss': Discriminator_loss,
            'D_support_loss': D_support_loss,
            'D_main_loss': D_main_loss,
        }

    def train(self):
        # Initialize the generator by running a dummy forward pass
        dummy_input = tf.zeros(
            [1, self.img_size[0], self.img_size[1], self.img_ch])
        self.gen(dummy_input, True)
        self.disc_support(dummy_input)
        self.disc_main(dummy_input)

        show_all_variables(self.gen, self.disc_support, self.disc_main)

        # Data loading
        real_dataset = self.real_generator.load_images()
        anime_dataset = self.anime_image_generator.load_images()
        anime_smooth_dataset = self.anime_smooth_generator.load_images()

        real_iter = iter(real_dataset)
        anime_iter = iter(anime_dataset)
        smooth_iter = iter(anime_smooth_dataset)

        # Restore checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            print(" [!] Load failed...")

        # Training loop
        steps = int(self.dataset_num / self.batch_size)
        for epoch in range(start_epoch, self.epoch):
            for idx in range(steps):
                start_time = time.time()

                real_photo = next(real_iter)
                anime_ = next(anime_iter)
                anime_smooth_ = next(smooth_iter)

                real_photo_img = real_photo[0]
                photo_superpixel = real_photo[1]
                anime_img = anime_[0]
                anime_smooth_img = anime_smooth_[0]

                """ pre-training G """
                if epoch < self.init_G_epoch:
                    init_loss = self.pretrain_G_step(real_photo_img)
                    step_time = time.time() - start_time
                    print("Epoch: %3d, Step: %5d / %5d, time: %.3fs, ETA: %.2fs, Pre_train_G_loss: %.6f" % (
                        epoch, idx, steps, step_time, step_time*(steps-idx+1), init_loss.numpy()))

                else:
                    """style transfer"""
                    # output fake image (forward pass for superpixel/NLMean processing)
                    generated_s, generated_m, generated = self.compute_generator_output(
                        real_photo_img, True)
                    inter_out_s = generated_s.numpy()
                    inter_out = generated.numpy()

                    superpixel_batch = self.get_seg(inter_out)
                    fake_NLMean_batch = self.get_NLMean_l0(inter_out_s)

                    fake_superpixel = tf.constant(superpixel_batch)
                    fake_NLMean_l0 = tf.constant(fake_NLMean_batch)

                    """ Update G """
                    g_results = self.train_G_step(
                        real_photo_img, photo_superpixel, anime_img, anime_smooth_img,
                        fake_superpixel, fake_NLMean_l0)

                    """ Update D """
                    d_results = self.train_D_step(
                        real_photo_img, photo_superpixel, anime_img, anime_smooth_img,
                        fake_superpixel, fake_NLMean_l0)

                    step_time = time.time() - start_time
                    G_loss = g_results['G_loss'].numpy()
                    D_loss = d_results['D_loss'].numpy()
                    info = f'Epoch: {epoch:3d}, Step: {idx:5d} /{steps:5d}, time: {step_time:.3f}s, ETA: {step_time*(steps-idx+1):.2f}s, ' + \
                           f'D_loss:{D_loss:.3f} ~ G_loss: {G_loss:.3f} || ' + \
                           f'G_support_loss: {g_results["G_support_loss"].numpy():.6f}, g_s_loss: {g_results["g_adv_loss"].numpy():.6f}, con_loss: {g_results["con_loss"].numpy():.6f}, rs_loss: {g_results["rs_loss"].numpy():.6f}, sty_loss: {g_results["sty_loss"].numpy():.6f}, s22: {g_results["s22"].numpy():.6f}, s33: {g_results["s33"].numpy():.6f}, s44: {g_results["s44"].numpy():.6f}, color_loss: {g_results["color_loss"].numpy():.6f}, tv_loss: {g_results["tv_loss"].numpy():.6f} ~ D_support_loss: {d_results["D_support_loss"].numpy():.6f} || ' + \
                           f'G_main_loss: {g_results["G_main_loss"].numpy():.6f}, g_m_loss: {g_results["g_m_loss"].numpy():.6f}, p0_loss: {g_results["p0_loss"].numpy():.6f}, p4_loss: {g_results["p4_loss"].numpy():.6f}, tv_loss_m: {g_results["tv_loss_m"].numpy():.6f} ~ D_main_loss: {d_results["D_main_loss"].numpy():.6f}'
                    print(info)

            # Save checkpoint
            if (epoch + 1) >= self.init_G_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            if (epoch + 1) >= self.init_G_epoch:
                """ Result Image """
                val_files = glob(os.path.join(self.dataset, 'val', '*.*'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                for i, sample_file in enumerate(val_files):
                    print('val: ' + str(i) + sample_file)
                    sample_image = np.asarray(
                        load_test_data(sample_file, self.img_size))

                    val_generated_s, val_generated_m, val_generated = self.compute_generator_output(
                        sample_image, False)

                    save_images(sample_image, save_path +
                                '{:03d}_a.jpg'.format(i))
                    save_images(val_generated.numpy(), save_path +
                                '{:03d}_b.jpg'.format(i))
                    save_images(val_generated_s.numpy(),
                                save_path+'{:03d}_c.jpg'.format(i))
                    save_images(val_generated_m.numpy(),
                                save_path+'{:03d}_d.jpg'.format(i))

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def get_seg(self, batch_image):
        def get_superpixel(image):
            image = (image + 1.) * 127.5
            image = np.clip(image, 0, 255).astype(
                np.uint8)  # [-1. ,1.] ~ [0, 255]
            image_seg = segmentation.felzenszwalb(
                image, scale=5, sigma=0.8, min_size=50)
            image = color.label2rgb(
                image_seg, image,  bg_label=-1, kind='avg').astype(np.float32)
            image = image / 127.5 - 1.0
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(
            delayed(get_superpixel)(image) for image in batch_image)
        return np.array(batch_out)

    def get_simple_superpixel(self, batch_image, seg_num=200):
        def process_slic(image):
            seg_label = segmentation.slic(
                image, n_segments=seg_num, sigma=1, start_label=0, compactness=10, convert2lab=True)
            image = color.label2rgb(seg_label, image, bg_label=-1, kind='avg')
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(
            delayed(process_slic)(image)for image in batch_image)
        return np.array(batch_out)

    def get_NLMean_l0(self, batch_image, ):
        def process_revision(image):
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            image = cv2.fastNlMeansDenoisingColored(image, None, 5, 6, 5, 7)
            image = L0Smoothing(image/255, 0.005).astype(np.float32) * 2. - 1.
            return image.clip(-1., 1.)
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(
            delayed(process_revision)(image) for image in batch_image)
        return np.array(batch_out)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint.write(
            os.path.join(checkpoint_dir, self.model_name + '.ckpt-' + str(step)))
        print(f" [*] Saved checkpoint at step {step}")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest:
            try:
                self.checkpoint.read(latest).expect_partial()
                # Extract step number from the checkpoint path
                counter = int(latest.split('-')[-1])
                print(" [*] Success to read {}".format(latest))
                return True, counter
            except Exception as e:
                print(f" [*] Failed to load checkpoint: {e}")
                return False, 0
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def to_lab(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        x = (x + 1.0) / 2.0
        x = rgb_to_lab(x)
        y = tf.concat([tf.expand_dims(x[:, :, :, 0] / 100., -1), tf.expand_dims(
            (x[:, :, :, 1]+128.)/255., -1), tf.expand_dims((x[:, :, :, 2]+128.)/255., -1)], axis=-1)
        return y
