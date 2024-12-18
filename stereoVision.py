import numpy as np
import cv2
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage import io, img_as_float32
import matplotlib.pyplot as plt

def compute_disparity(left_image, right_image, block_size=5, max_disparity=64):
    h, w = left_image.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    half_block = block_size // 2

    for y in tqdm(range(half_block, h - half_block), desc="Computing Disparity"):
        for x in range(half_block, w - half_block):
            best_offset = 0
            min_ssd = float('inf')

            for offset in range(max_disparity):
                if x - offset - half_block < 0:
                    break

                left_block = left_image[y - half_block:y + half_block + 1, x - half_block:x + half_block + 1]
                right_block = right_image[y - half_block:y + half_block + 1, x - offset - half_block:x - offset + half_block + 1]

                ssd = np.sum((left_block - right_block) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = offset

            disparity_map[y, x] = best_offset

    return disparity_map

def create_portrait_mode(image1_file, image2_file, blur_intensity=71, block_size=10, max_disparity=60):
    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)

    scale_factor = 0.2
    image1_gray = cv2.resize(image1_gray, (0, 0), fx=scale_factor, fy=scale_factor)
    image2_gray = cv2.resize(image2_gray, (0, 0), fx=scale_factor, fy=scale_factor)

    disparity = compute_disparity(image1_gray, image2_gray, block_size=block_size, max_disparity=max_disparity)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    _, initial_mask = cv2.threshold(disparity_normalized, 0.3, 1.0, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(initial_mask.astype(np.uint8))
    refined_mask = np.zeros_like(initial_mask, dtype=np.float32)

    for label in range(1, num_labels):
        region = (labels == label)
        if np.sum(region) > 500: 
            refined_mask[region] = 1

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    refined_mask = cv2.resize(refined_mask, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
    refined_mask = refined_mask.astype(np.float32)

    blurred_image = cv2.GaussianBlur(image1, (blur_intensity, blur_intensity), 0)
    portrait_image = (refined_mask[..., np.newaxis] * image1 + (1 - refined_mask[..., np.newaxis]) * blurred_image)

    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.title("Disparity Map")
    plt.imshow(disparity_normalized)

    # plt.subplot(1, 3, 2)
    # plt.title("Initial Mask")
    # plt.imshow(initial_mask, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Refined Mask")
    plt.imshow(refined_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Portrait Effect")
    plt.imshow(portrait_image)

    plt.show()

def load_image(image_file):

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_as_float32(image)

def main():
    left_image_path = 'images/view1.png'
    right_image_path = 'images/view5.png'

    image1 = load_image(left_image_path)
    image2 = load_image(right_image_path)
    
    create_portrait_mode(image1, image2)

if __name__ == "__main__":
    main()
