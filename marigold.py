import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt

# Load marigold depth estimation model
def load_depth_model():
    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Depth model loaded.")
    return model


# Predict the depth map using the model
def predict_depth(model, image):
    depth = model(image)
    depth_map = depth.prediction[0]  # Directly use the NumPy array output
    return depth_map

# Preprocess the input image?
def preprocess_image(image_path):
    image = diffusers.utils.load_image(image_path)
    return image

# Calculate the threshold for separating foreground and background
def calculate_threshold(depth_map, percentile=60):
    threshold = np.percentile(depth_map, percentile)
    print(f"Dynamic threshold (percentile {percentile}): {threshold:.3f}")
    return threshold

# Generate a binary mask separating foreground and background
def generate_foreground_mask(depth_map, threshold):
    # Foreground: depth values < threshold
    mask = (depth_map < threshold).astype(np.uint8) * 255
    return mask

# Refine the mask with Gaussian blur and morphological operations
def refine_mask(mask):
    # Smooth mask edges using Gaussian blur
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Clean up using morphological closing
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    
    return refined_mask

# Combine the blurred background and the sharp foreground
def combine_foreground_background(image, mask):
    """
    Combine the sharp foreground and blurred background with a smooth transition.
    """
    # Feather the mask to create a smooth transition
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)  # Feathering the mask edges
    
    # Normalize the blurred mask to range [0, 1] for alpha blending
    alpha = blurred_mask.astype(np.float32) / 255.0
    
    # Sharpen the foreground
    sharpening_kernel = np.array([[0, -1,  0],
                                  [-1, 5, -1],
                                  [0, -1,  0]])
    sharpened_foreground = cv2.filter2D(image, -1, sharpening_kernel)
    
    # Extract the sharp foreground and blurred background
    foreground = sharpened_foreground.astype(np.float32)
    blurred_background = cv2.GaussianBlur(image, (151, 151), 0).astype(np.float32)
    
    # Perform alpha blending: Combine foreground and background smoothly
    combined_image = (foreground * alpha[..., None] + blurred_background * (1 - alpha[..., None])).astype(np.uint8)
    
    return combined_image

# Visualize a mask for debugging
def visualize_mask(mask, title="Mask Visualization"):
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Visualize the depth map for debugging or visualization purposes
def visualize_depth_map(depth_map, output_path="depth_map_visualization.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap="plasma")
    plt.colorbar()
    plt.title("Depth Map Visualization")
    plt.savefig(output_path)
    plt.close()
    print(f"Depth map visualization saved to {output_path}")

def smooth_mask(mask, blur_radius=15):
    # Apply Gaussian Blur to mask to smooth transition
    smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_radius, blur_radius), 0)
    smoothed_mask = np.clip(smoothed_mask, 0, 1)  # Ensure the mask stays between 0 and 1
    return smoothed_mask

# Apply portrait mode effect
def apply_portrait_mode(image_path, output_path):
    # Step 1: Load the input image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Step 2: Load the depth estimation model
    depth_model = load_depth_model()

    # Step 3: Preprocess the image for the model
    input_image = preprocess_image(image_path)

    # Step 4: Predict the depth map
    depth_map = predict_depth(depth_model, input_image)

    # Step 5: Resize depth map to match original image dimensions
    depth_map_resized = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]))

    # Visualize the depth map (optional)
    visualize_depth_map(depth_map_resized)

    # Step 6: Threshold to create a binary foreground mask
    threshold = calculate_threshold(depth_map, percentile=60)
    foreground_mask = generate_foreground_mask(depth_map, threshold)
    refined_mask = refine_mask(foreground_mask)

    # Debugging: Visualize the mask (optional)
    visualize_mask(refined_mask, "Refined Foreground Mask")

    # Step 8: Combine sharp foreground and blurred background
    portrait_image = combine_foreground_background(original_image, refined_mask)

    # Step 11: Save the resulting image
    cv2.imwrite(output_path, portrait_image)

    print(f"Portrait mode image saved to {output_path}")

# Example usage
input_image_path = "images/IMG_8967.jpeg"  # Replace with the path to your input image
output_image_path = "portrait_mode_output.jpg"

apply_portrait_mode(input_image_path, output_image_path)
