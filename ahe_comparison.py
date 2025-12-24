import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Load and prepare image ---
img = plt.imread('couple.tiff')
if img.max() <= 1:
    img = (img * 255).astype(np.uint8)
else:
    img = img.astype(np.uint8)


# ============================================================
# Helper Functions
# ============================================================

def histogram_equalization(block):
    """Perform histogram equalization on a local image block."""
    M, N = block.shape
    hist, _ = np.histogram(block.flatten(), 256, [0, 256])
    p_r = hist / (M * N)  # normalized histogram (probabilities)

    # Compute CDF
    cdf = np.zeros(256)
    for k in range(256):
        cdf[k] = np.sum(p_r[:k + 1])

    # Normalize to [0,255]
    s_k = np.round(255 * cdf).astype(np.uint8)
    return s_k[block]


def entropy_manual(img):
    """Compute image entropy based on histogram probabilities."""
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    p = hist / np.sum(hist)
    p = p[p > 0]  # avoid log(0)
    return -np.sum(p * np.log2(p))


def psnr_manual(original, equalized):
    """Compute PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = np.mean((original.astype(np.float64) - equalized.astype(np.float64)) ** 2)
    return 10 * np.log10((255 ** 2) / mse)


# ============================================================
# AHE Approaches
# ============================================================

def tiling_ahe(img, block_size):
    """Adaptive Histogram Equalization using Tiling (non-overlapping blocks)."""
    h, w = img.shape
    out = np.zeros_like(img)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j + block_size]
            eq_block = histogram_equalization(block)
            out[i:i + block_size, j:j + block_size] = eq_block

    return out


def sliding_ahe(img, window_size):
    """Adaptive Histogram Equalization using Sliding Window (overlapping)."""
    h, w = img.shape
    pad = window_size // 2
    padded = np.pad(img, pad, mode='reflect')
    out = np.zeros_like(img)

    step = window_size // 4  # controls overlap
    for i in range(0, h, step):
        for j in range(0, w, step):
            block = padded[i:i + window_size, j:j + window_size]

            # Histogram equalization inside the sliding window
            hist, _ = np.histogram(block.flatten(), 256, [0, 256])
            p_r = hist / block.size
            cdf = np.cumsum(p_r)
            s_k = np.round(255 * cdf).astype(np.uint8)

            out[i:i + step, j:j + step] = s_k[img[i:i + step, j:j + step]]

    return out.astype(np.uint8)


# ============================================================
# Comparison and Visualization
# ============================================================

sizes = [64, 32, 16]
tiling_entropies, tiling_psnrs = [], []
sliding_entropies, sliding_psnrs = [], []

plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# --- AHE with Tiling ---
for idx, size in enumerate(sizes, start=2):
    out_tiling = tiling_ahe(img, size)
    ent_t = entropy_manual(out_tiling)
    psnr_t = psnr_manual(img, out_tiling)
    tiling_entropies.append(ent_t)
    tiling_psnrs.append(psnr_t)

    plt.subplot(2, 4, idx)
    plt.imshow(out_tiling, cmap='gray')
    plt.title(f'Tiling {size}×{size}')
    plt.axis('off')

    print(f'Tiling {size}×{size}: Entropy = {ent_t:.4f}, PSNR = {psnr_t:.4f} dB')


# --- AHE with Sliding Window ---
for idx, size in enumerate(sizes, start=5):
    out_sliding = sliding_ahe(img, size)
    ent_s = entropy_manual(out_sliding)
    psnr_s = psnr_manual(img, out_sliding)
    sliding_entropies.append(ent_s)
    sliding_psnrs.append(psnr_s)

    plt.subplot(2, 4, idx)
    plt.imshow(out_sliding, cmap='gray')
    plt.title(f'Sliding {size}×{size}')
    plt.axis('off')

    print(f'Sliding {size}×{size}: Entropy = {ent_s:.4f}, PSNR = {psnr_s:.4f} dB')

plt.tight_layout()
plt.show()


# ============================================================
# Entropy and PSNR Plots
# ============================================================

plt.figure(figsize=(12, 5))

# --- Entropy Plot ---
plt.subplot(1, 2, 1)
plt.plot(sizes, tiling_entropies, 'o-b', label='Tiling')
plt.plot(sizes, sliding_entropies, 'o-r', label='Sliding')
plt.xlabel('Window/Block Size')
plt.ylabel('Entropy')
plt.title('Entropy vs Window Size')
plt.legend()
plt.grid(True)

# --- PSNR Plot ---
plt.subplot(1, 2, 2)
plt.plot(sizes, tiling_psnrs, 'o-b', label='Tiling')
plt.plot(sizes, sliding_psnrs, 'o-r', label='Sliding')
plt.xlabel('Window/Block Size')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Window Size')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
