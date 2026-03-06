"""
Export AnimeGANv3 Generator from TF2 checkpoint to ONNX format.

Usage:
    python export_onnx.py --checkpoint_dir checkpoint/shinkai
    python export_onnx.py --checkpoint_dir checkpoint/shinkai --output generator.onnx
"""
import argparse
import os
import numpy as np
import tensorflow as tf

from net.generator import Generator
from net.discriminator import Discriminator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export AnimeGANv3 Generator to ONNX')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/shinkai',
                        help='Directory containing the TF2 checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path (default: <checkpoint_dir>/generator.onnx)')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size (will be used for H and W)')
    return parser.parse_args()


def export(checkpoint_dir, output_path, input_size):
    print(f"[1/4] Building models (to match checkpoint structure)...")

    # Build all models to match checkpoint structure
    gen = Generator(name='generator')
    disc_support = Discriminator(sn=True, ch=32, name='discriminator')
    disc_main = Discriminator(sn=True, ch=32, name='discriminator_main')

    # Initialize with dummy input so variables are created
    dummy = tf.zeros([1, input_size, input_size, 3])
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

    print(f"[2/4] Loading checkpoint from {checkpoint_dir}...")
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        checkpoint.read(latest).expect_partial()
        print(f"  [*] Success to read {latest}")
    else:
        print("  [!] Failed to find a checkpoint")
        return

    # Create a concrete function for the generator (inference only)
    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32)])
    def generator_inference(x):
        fake_s, fake_m = gen(x, False)
        return fake_s, fake_m

    print(f"[3/4] Saving as SavedModel...")
    saved_model_dir = os.path.join(checkpoint_dir, 'saved_model_tmp')
    tf.saved_model.save(gen, saved_model_dir, signatures={
        'serving_default': generator_inference
    })

    print(f"[4/4] Converting SavedModel to ONNX...")
    # Use tf2onnx command line (more reliable than Python API for complex models)
    import subprocess
    cmd = [
        'python', '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', output_path,
        '--opset', '13',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [!] tf2onnx conversion failed:")
        print(result.stderr)
        return

    print(f"\n[OK] ONNX model saved to: {output_path}")
    print(
        f"     Model size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    # Clean up temporary SavedModel
    import shutil
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    # Quick verification with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        test_input = np.random.randn(
            1, input_size, input_size, 3).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: test_input})
        print(
            f"  Verification passed! Output shapes: {[o.shape for o in outputs]}")
    except ImportError:
        print("  [!] onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"  [!] Verification failed: {e}")


if __name__ == '__main__':
    args = parse_args()
    output = args.output or os.path.join(args.checkpoint_dir, 'generator.onnx')
    export(args.checkpoint_dir, output, args.input_size)
