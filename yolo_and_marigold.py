import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
import os

# Load YOLOv5 for object detection
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("YOLOv5 model loaded.")
    return model

# Load Marigold depth estimation model
def load_depth_model():
    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Marigold depth model loaded.")
    return model

# Predict depth map using Marigold model
def predict_depth(model, image):
    depth = model(image)
    depth_map = depth.prediction[0]
    return depth_map

# Preprocess input image for Marigold
def preprocess_image(image_path):
    image = diffusers.utils.load_image(image_path)
    return image

# Calculate average depth in the YOLO-detected bounding box
def calculate_average_depth(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    cropped_region = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(cropped_region)
    print(f"Average depth in bounding box: {avg_depth:.3f}")
    return avg_depth

# Generate binary mask separating foreground and background
def generate_foreground_mask(depth_map, threshold):
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

# Combine sharpened foreground and blurred background
def combine_foreground_background(image, mask):
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = blurred_mask.astype(np.float32) / 255.0
    
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_foreground = cv2.filter2D(image, -1, sharpening_kernel)
    blurred_background = cv2.GaussianBlur(image, (51, 51), 0)
    
    foreground = sharp_foreground.astype(np.float32)
    background = blurred_background.astype(np.float32)
    combined = (foreground * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)
    
    return combined

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

def draw_green_box(image_path, bbox, output_path):
    """
    Draws a green bounding box on the image and saves the result.
    Args:
        image_path (str): Path to the input image.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        output_path (str): Path to save the output image.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Draw the green box (BGR: (0, 255, 0))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thickness = 3

    # Save the image with the green box
    cv2.imwrite(output_path, image)
    print(f"Green box image saved to {output_path}")


# Main function to apply combined YOLO and Marigold processing
def apply_portrait_mode(image_path, output_image_path, depth_output_path, mask_output_path, box_output_path):

    # Load models
    yolo_model = load_yolo_model()
    depth_model = load_depth_model()

    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    input_image = preprocess_image(image_path)

    # Predict depth map
    depth_map = predict_depth(depth_model, input_image)
    depth_map_resized = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]))
    visualize_depth_map(depth_map_resized, depth_output_path)

    # Use YOLO to detect person
    results = yolo_model(image_path)
    detections = results.pandas().xyxy[0]
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        print(f"Object detected at: {x1}, {y1}, {x2}, {y2}")

        # Draw green box and save the result
        draw_green_box(image_path, (x1, y1, x2, y2), box_output_path)
        
        # Calculate average depth in bounding box
        avg_depth = calculate_average_depth(depth_map_resized, (x1, y1, x2, y2))
        
        # Generate mask using average depth
        foreground_mask = generate_foreground_mask(depth_map_resized, avg_depth)
        refined_mask = refine_mask(foreground_mask)

        # Debugging: Visualize the mask (optional)
        visualize_mask(refined_mask, mask_output_path)

        # Combine sharp foreground and blurred background
        combined_image = combine_foreground_background(original_image, refined_mask)
        cv2.imwrite(output_image_path, combined_image)
        print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nGreen Box -> {box_output_path},\nPortrait -> {output_image_path}")

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
        box_output_path = os.path.join(image_output_folder, f"{i+1}_box.png")

        try:
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            
            # Call the apply_portrait_mode function with all paths
            apply_portrait_mode(input_image_path, portrait_output_path, depth_output_path, mask_output_path, box_output_path)

            print(f"Saved outputs for {image_file} to {image_output_folder}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
input_folder = "marigold_dataset"  # Replace with your input folder path
output_folder = "yolo_outputs"  # Replace with your output folder path

process_image_folder(input_folder, output_folder)
