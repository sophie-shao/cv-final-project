import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load marigold depth estimation model
def load_depth_model():
    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Depth model loaded.")
    return model

# Predict the depth map
def predict_depth(model, image):
    depth = model(image)
    depth_map = depth.prediction[0]
    return depth_map

# Preprocess the input image?
def preprocess_image(image_path):
    image = diffusers.utils.load_image(image_path)
    return image

def enhance_subject(image, mask):
    # Extract the subject using mask 
    subject_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8)) 
    
    # Apply enhancements like contrast, sharpness, etc., to the subject (foreground)    
    enhanced_subject = cv2.convertScaleAbs(subject_image, alpha=1.05, beta=20)  # enhance brightness and contrast
    
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  #sharpening filter
    enhanced_subject = cv2.filter2D(enhanced_subject, -1, kernel)
    return enhanced_subject

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

    # Step 6: Separate foreground and background
    threshold = np.percentile(depth_map_resized, 30)  # Adjust as needed
    foreground_mask = depth_map_resized < threshold

    # Step 7: Apply Gaussian blur to the background
    blurred_image = cv2.GaussianBlur(original_image, (45, 45), 100)

    # Step 8: Enhance subject (foreground) and make edges smooth
    enhanced_image = enhance_subject(original_image, foreground_mask)
    smoothed_mask = smooth_mask(foreground_mask)

     # Step 9: Convert mask to 3 channels to apply on RGB image
    foreground_mask_3d = np.repeat(smoothed_mask[:, :, np.newaxis], 3, axis=2)

    # Step 10: Combine the sharp foreground with the blurred background
    portrait_mode = (foreground_mask_3d * enhanced_image + (1 - foreground_mask_3d) * blurred_image).astype(np.uint8)

    # Step 9: Save the resulting image
    cv2.imwrite(output_path, portrait_mode)
    print(f"Portrait mode image saved to {output_path}")

# Example usage
input_image_path = "images/sophie.jpg"  # Replace with the path to your input image
output_image_path = "portrait_mode_output.jpg"

apply_portrait_mode(input_image_path, output_image_path)
