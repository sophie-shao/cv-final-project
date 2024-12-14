import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the Marigold depth estimation model
def load_depth_model():
    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Depth model loaded.")
    return model

# Predict the depth map usingthe model
def predict_depth(model, image):
    depth = model(image)
    depth_map = depth.prediction[0]
    return depth_map

# Preprocess the input image (resize and normalize as needed by the model)
def preprocess_image(image_path):
    image = diffusers.utils.load_image(image_path)
    return image

def enhance_subject(image, mask):
    """Enhance the subject based on the binary mask"""
    # We will apply enhancements like contrast, sharpness, etc., to the subject (foreground)
    subject_image = cv2.bitwise_and(image, image, mask=mask)  # Extract the subject using the mask
    enhanced_subject = cv2.convertScaleAbs(subject_image, alpha=1.2, beta=50)  # Example of contrast and brightness enhancement
    return enhanced_subject

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

    # Step 6: Separate foreground and background
    threshold = np.percentile(depth_map_resized, 30)  # Adjust as needed
    foreground_mask = depth_map_resized < threshold

    # Convert mask to 3 channels to apply on RGB image
    foreground_mask_3d = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)

    # Step 7: Apply Gaussian blur to the background
    blurred_image = cv2.GaussianBlur(original_image, (15, 15), 0)

    enhanced_image = enhance_subject(original_image, foreground_mask)

    # Step 8: Combine the sharp foreground with the blurred background
    portrait_mode_image = np.where(foreground_mask_3d, enhanced_image, blurred_image)

    # Step 9: Save the resulting image
    cv2.imwrite(output_path, portrait_mode_image)
    print(f"Portrait mode image saved to {output_path}")

# Example usage
input_image_path = "images/sophie.jpg"  # Replace with the path to your input image
output_image_path = "portrait_mode_output.jpg"

apply_portrait_mode(input_image_path, output_image_path)
