import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import io, img_as_float32
import matplotlib.pyplot as plt

def compute_disparity(left_image, right_image, block_size=5, max_disparity=64):
    """
    Compute disparity map using basic block matching.
    
    Parameters:
        left_image: Grayscale left stereo image.
        right_image: Grayscale right stereo image.
        block_size: Size of the matching block.
        max_disparity: Maximum disparity to search.

    Returns:
        Disparity map as a 2D numpy array.
    """
    h, w = left_image.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    half_block = block_size // 2

    for y in range(half_block, h - half_block):
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

def create_portrait_mode(image1_file, image2_file, blur_intensity=15, block_size=5, max_disparity=64):
    # Load and preprocess stereo images
    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))
    image1 = rgb2gray(image1)
    image2 = rgb2gray(image2)

    # Scale images for faster computation
    scale_factor = 0.5
    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)

    # Compute disparity map
    disparity = compute_disparity(image1, image2, block_size=block_size, max_disparity=max_disparity)

    # Normalize disparity for visualization and processing
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Initial mask creation based on disparity threshold
    _, initial_mask = cv2.threshold(disparity_normalized, 0.3, 1.0, cv2.THRESH_BINARY)
    initial_mask = (initial_mask * 255).astype(np.uint8)

    # Apply connected components to refine the mask
    num_labels, labels = cv2.connectedComponents(initial_mask)
    refined_mask = np.zeros_like(initial_mask, dtype=np.float32)

    # Filter small regions to refine the subject mask
    for label in range(1, num_labels):  # Skip label 0 (background)
        region = (labels == label)
        if np.sum(region) > 500:  # Adjust minimum size for subject regions
            refined_mask[region] = 1

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # Resize mask to original image size
    refined_mask = cv2.resize(refined_mask, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply blurring to the background
    blurred_image = cv2.GaussianBlur(image1, (blur_intensity, blur_intensity), 0)

    # Combine layers
    portrait_image = refined_mask * image1 + (1 - refined_mask) * blurred_image

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Disparity Map")
    plt.imshow(disparity_normalized, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Refined Mask")
    plt.imshow(refined_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Portrait Effect")
    plt.imshow(portrait_image, cmap='gray')

    plt.show()


def main():
    # Replace these with the actual file paths to your stereo images
    left_image_path = 'images/robot_left.png'
    right_image_path = 'images/robot_right.png'
    
    create_portrait_mode(left_image_path, right_image_path)

if __name__ == "__main__":
    main()
