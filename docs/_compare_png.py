#!/usr/bin/env python3
"""
PNG Comparison Tool for CI

This script compares two PNG files to check if they are pixel-by-pixel identical.
It's designed for use in CI workflows to verify that generated images match
expected reference images, accounting for potential binary differences that
don't affect the visual content (e.g., different compression settings).

The script loads both images as numpy arrays and performs a direct comparison.
If the images have different dimensions or any pixels differ, it returns an error.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_as_array(image_path):
    """Load image and convert to numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def compare_images_pixel_perfect(file1, file2):
    """
    Compare two PNG files for pixel-perfect equality.

    Args:
        file1: Path to first PNG file
        file2: Path to second PNG file

    Returns:
        bool: True if images are identical, False otherwise
    """
    # Load images
    img1 = load_image_as_array(file1)
    img2 = load_image_as_array(file2)

    if img1 is None or img2 is None:
        return False

    # Check if shapes are identical
    if img1.shape != img2.shape:
        print(f"Images have different dimensions: {img1.shape} vs {img2.shape}")
        return False

    # Check if arrays are identical
    if np.array_equal(img1, img2):
        return True
    else:
        # Calculate some metrics for debugging
        diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
        diff_pixels = np.sum(diff > 0)
        total_pixels = img1.size
        diff_percentage = (diff_pixels / total_pixels) * 100

        print(
            f"Images differ: {diff_pixels} / {total_pixels} "
            f"pixels ({diff_percentage:.2f}%)"
        )
        print(f"Maximum difference: {np.max(diff):.2f}")
        return False


def main():
    """Main function for CLI usage."""
    if len(sys.argv) != 3:
        print("Usage: python _compare_png.py <file1> <file2>")
        print("Compares two PNG files for pixel-perfect equality.")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    # Check if files exist
    if not Path(file1).exists():
        print(f"Error: {file1} does not exist")
        sys.exit(1)
    if not Path(file2).exists():
        print(f"Error: {file2} does not exist")
        sys.exit(1)

    # Compare images
    if compare_images_pixel_perfect(file1, file2):
        print("✓ Images are identical")
        sys.exit(0)
    else:
        print("✗ Images differ")
        sys.exit(1)


if __name__ == "__main__":
    main()
