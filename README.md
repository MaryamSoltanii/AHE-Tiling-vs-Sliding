# Adaptive Histogram Equalization (AHE) - Tiling vs Sliding Window Comparison

## Overview
This repository implements and compares two approaches of Adaptive Histogram Equalization (AHE) for image enhancement:
- **Tiling Approach**: Divides the image into non-overlapping blocks and applies histogram equalization to each block
- **Sliding Window Approach**: Uses overlapping windows to compute local histograms for each pixel neighborhood

## Features
- Implementation of both Tiling and Sliding Window AHE methods
- Quantitative evaluation using Entropy and PSNR metrics
- Visualization of enhanced images and performance graphs
- Comparison of different window/block sizes (16x16, 32x32, 64x64)

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- OpenCV (cv2)

Install dependencies:
```bash
pip install numpy matplotlib opencv-python
