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

def create_portrait_mode(image1_file, image2_file, blur_intensity=45, block_size=5, max_disparity=64):
    # Load and preprocess stereo images
    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))
    image1 = rgb2gray(image1)
    image2 = rgb2gray(image2)

    # Scale images for faster computation
    scale_factor = 0.2
    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)

    # Compute disparity map
    disparity = compute_disparity(image1, image2, block_size=block_size, max_disparity=max_disparity)

    # Normalize disparity for visualization and processing
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Segment the image based on disparity
    _, mask = cv2.threshold(disparity_normalized, 0.5, 1.0, cv2.THRESH_BINARY)

    # Resize mask to original image size
    mask = mask.astype(np.float32)

    # Apply blurring to the background
    blurred_image = cv2.GaussianBlur(image1, (blur_intensity, blur_intensity), 0)

    # Combine layers
    portrait_image = mask * image1 + (1 - mask) * blurred_image

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Disparity Map")
    plt.imshow(disparity_normalized, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Portrait Effect")
    plt.imshow(portrait_image, cmap='gray')

    plt.show()

def main():
    # Replace these with the actual file paths to your stereo images
    left_image_path = 'images/eye2eye1.jpg'
    right_image_path = 'images/eye2eye2.jpg'
    
    create_portrait_mode(left_image_path, right_image_path)

if __name__ == "__main__":
    main()
