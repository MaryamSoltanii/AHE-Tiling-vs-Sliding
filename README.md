# Adaptive Histogram Equalization (AHE) — Tiling vs Sliding Window

This repository implements **Adaptive Histogram Equalization (AHE)** using two different approaches:

1. **Tiling Approach** — The image is divided into equally-sized blocks, and histogram equalization is applied locally to each.
2. **Sliding Window Approach** — A window slides across the image with overlap, performing local equalization in each region.

The code compares both methods in terms of **Entropy** and **PSNR**, and visualizes:
- The equalized images for multiple window/block sizes
- Plots of Entropy and PSNR vs. window size

---

## 🚀 Features
- Manual implementation of histogram equalization (without OpenCV’s built-in `cv2.equalizeHist`)
- Two local AHE variants (tiling & sliding window)
- Quantitative comparison using Entropy and PSNR
- Visualization using Matplotlib

---

## 🧰 Requirements
```bash
pip install numpy matplotlib
