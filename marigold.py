import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
import os

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
def visualize_mask(mask, output_path="mask_visualization.png"):
    plt.imshow(mask, cmap="gray")
    plt.title("Mask Visualization")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    print(f"Mask visualization saved to {output_path}")

# Visualize the depth map for debugging or visualization purposes
def visualize_depth_map(depth_map, output_path="depth_map_visualization.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap="plasma")
    plt.colorbar()
    plt.title("Depth Map Visualization")
    plt.savefig(output_path)
    plt.close()
    print(f"Depth map visualization saved to {output_path}")

# Apply portrait mode effect
def apply_portrait_mode(image_path, output_image_path, depth_output_path, mask_output_path):

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

    # Step 6: Save the depth map visualization
    visualize_depth_map(depth_map_resized, depth_output_path)

    # Step 7: Threshold to create a binary foreground mask
    threshold = calculate_threshold(depth_map, percentile=60)
    foreground_mask = generate_foreground_mask(depth_map, threshold)
    refined_mask = refine_mask(foreground_mask)

    # Step 8: Save the mask visualization
    visualize_mask(refined_mask, mask_output_path)

    # Step 9: Combine sharp foreground and blurred background
    portrait_image = combine_foreground_background(original_image, refined_mask)

    # Step 10: Save the resulting image
    cv2.imwrite(output_image_path, portrait_image)

    print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nPortrait -> {output_image_path}")

# Process an entire folder of images
def process_image_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Get all image paths in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpeg', '.jpg', '.png'))]

    print(f"Found {len(image_files)} images in {input_folder}. Processing...")

    # Loop through each image
    for i, image_file in enumerate(image_files):
        input_image_path = os.path.join(input_folder, image_file)

        # Create output subfolder for each image group
        image_output_folder = os.path.join(output_folder, f"output_{i+1}")
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        # Define output paths for depth map, mask, and portrait image
        depth_output_path = os.path.join(image_output_folder, f"{i+1}_depth_map.png")
        mask_output_path = os.path.join(image_output_folder, f"{i+1}_mask.png")
        portrait_output_path = os.path.join(image_output_folder, f"{i+1}_portrait.png")

        try:
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            
            # Call the apply_portrait_mode function with all paths
            apply_portrait_mode(input_image_path, portrait_output_path, depth_output_path, mask_output_path)

            print(f"Saved outputs for {image_file} to {image_output_folder}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
input_folder = "marigold_dataset"  # Replace with your input folder path
output_folder = "marigold_outputs"  # Replace with your output folder path

process_image_folder(input_folder, output_folder)