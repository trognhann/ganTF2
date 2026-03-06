"""
Test AnimeGANv3 using ONNX model (lightweight, no TensorFlow needed at inference).

Usage:
    python test_onnx.py --model checkpoint/shinkai/generator.onnx --test_dir inputs/imgs
    python test_onnx.py --model checkpoint/shinkai/generator.onnx --test_dir inputs/imgs --max_size 512
"""
import argparse
import os
import time
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    exit(1)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, image_path, hw):
    """Save output image, rescaling from [-1,1] to [0,255]."""
    images = (images.squeeze() + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, (hw[1], hw[0]))
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))


def preprocessing(img, max_size=512):
    """Resize image so the longest side <= max_size, aligned to 8px."""
    h, w = img.shape[:2]
    ratio = min(max_size / max(h, w), 1.0)  # don't upscale
    new_h = max(256, int(h * ratio))
    new_w = max(256, int(w * ratio))
    # Align to 8
    new_h = new_h - new_h % 8
    new_w = new_w - new_w % 8
    img = cv2.resize(img, (new_w, new_h))
    return img / 127.5 - 1.0


def load_test_data(image_path, max_size=512):
    img0 = cv2.imread(image_path)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocessing(img, max_size)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, img0.shape[:2]


# ---------- NumPy Guided Filter (replaces TF version) ----------

def _diff_x(input_arr, r):
    left = input_arr[:, :, r:2*r+1]
    middle = input_arr[:, :, 2*r+1:] - input_arr[:, :, :-2*r-1]
    right = input_arr[:, :, -1:] - input_arr[:, :, -2*r-1:-r-1]
    return np.concatenate([left, middle, right], axis=2)


def _diff_y(input_arr, r):
    left = input_arr[:, :, :, r:2*r+1]
    middle = input_arr[:, :, :, 2*r+1:] - input_arr[:, :, :, :-2*r-1]
    right = input_arr[:, :, :, -1:] - input_arr[:, :, :, -2*r-1:-r-1]
    return np.concatenate([left, middle, right], axis=3)


def _box_filter(x, r):
    return _diff_y(np.cumsum(_diff_x(np.cumsum(x, axis=2), r), axis=3), r)


def guided_filter_np(x, y, r, eps=1e-1):
    """
    NumPy implementation of guided filter.
    x, y: NHWC format, values in [0, 1]
    """
    # Convert to NCHW
    x = np.transpose(x, [0, 3, 1, 2])
    y = np.transpose(y, [0, 3, 1, 2])

    N = _box_filter(np.ones((1, 1, x.shape[2], x.shape[3]), dtype=x.dtype), r)
    mean_x = _box_filter(x, r) / N
    mean_y = _box_filter(y, r) / N
    cov_xy = _box_filter(x * y, r) / N - mean_x * mean_y
    var_x = _box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = _box_filter(A, r) / N
    mean_b = _box_filter(b, r) / N

    output = mean_A * x + mean_b
    # Convert back to NHWC
    output = np.transpose(output, [0, 2, 3, 1])
    return output


def sigm_out_scale(x):
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def tanh_out_scale(x):
    return np.clip((x - 0.5) * 2.0, -1.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='AnimeGANv3 ONNX Test')
    parser.add_argument('--model', type=str, default='checkpoint/shinkai/generator.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--test_dir', type=str, default='inputs/imgs',
                        help='Directory of test photos')
    parser.add_argument('--save_dir', type=str, default='style_results/',
                        help='Directory to save results')
    parser.add_argument('--max_size', type=int, default=512,
                        help='Max size for longest side of input image (default: 512)')
    return parser.parse_args()


def test(model_path, save_dir, test_dir, max_size):
    result_dir = check_folder(save_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    if not test_files:
        print(f"No test images found in {test_dir}")
        return

    print(f"Loading ONNX model: {model_path}")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  Input: {input_name}, Outputs: {output_names}")
    print(f"  Max input size: {max_size}px")

    begin = time.time()
    for sample_file in tqdm(test_files):
        sample_image, scale = load_test_data(sample_file, max_size)
        print(
            f"  Processing {os.path.basename(sample_file)}: input shape {sample_image.shape}")

        # Run ONNX inference
        outputs = sess.run(None, {input_name: sample_image})
        test_s0 = outputs[0]  # support branch
        test_m = outputs[1]   # main branch

        # Apply guided filter (NumPy version)
        test_s1 = tanh_out_scale(
            guided_filter_np(sigm_out_scale(test_s0),
                             sigm_out_scale(test_s0), 2, 0.01)
        )

        # Save results (resize back to original size)
        basename = os.path.basename(sample_file)
        save_images(sample_image, os.path.join(
            result_dir, f'a_{basename}'), scale)
        save_images(test_s1, os.path.join(result_dir, f'b_{basename}'), scale)
        save_images(test_s0, os.path.join(result_dir, f'c_{basename}'), scale)
        save_images(test_m, os.path.join(result_dir, f'd_{basename}'), scale)

    end = time.time()
    print(f'\nTest completed!')
    print(f'Total time: {end - begin:.2f}s')
    print(f'Per image: {(end - begin) / len(test_files):.2f}s')
    print(f'Results saved to: {result_dir}')


if __name__ == '__main__':
    args = parse_args()
    test(args.model, args.save_dir, args.test_dir, args.max_size)
