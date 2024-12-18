import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
import os

input_folder = "marigold_dataset"
output_folder = "marigold_outputs" 

def load_depth_model():

    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Depth model loaded.")
    return model

def predict_depth(model, image):

    depth = model(image)
    depth_map = depth.prediction[0]
    return depth_map

def preprocess_image(image_path):

    image = diffusers.utils.load_image(image_path)
    return image

def calculate_threshold(depth_map, percentile=60):

    threshold = np.percentile(depth_map, percentile)
    print(f"Dynamic threshold (percentile {percentile}): {threshold:.3f}")
    return threshold

def generate_foreground_mask(depth_map, threshold):

    mask = (depth_map < threshold).astype(np.uint8) * 255
    return mask

def refine_mask(mask):
  
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    
    return refined_mask

def combine_foreground_background(image, mask):
  
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0) 
    alpha = blurred_mask.astype(np.float32) / 255.0
   
    sharpening_kernel = np.array([[0, -1,  0],
                                  [-1, 5, -1],
                                  [0, -1,  0]])
    sharpened_foreground = cv2.filter2D(image, -1, sharpening_kernel)
    
    foreground = sharpened_foreground.astype(np.float32)
    blurred_background = cv2.GaussianBlur(image, (151, 151), 0).astype(np.float32)
    
    combined_image = (foreground * alpha[..., None] + blurred_background * (1 - alpha[..., None])).astype(np.uint8)
    
    return combined_image


def apply_portrait_mode(image_path, output_image_path, depth_output_path, mask_output_path):

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")


    depth_model = load_depth_model()
    input_image = preprocess_image(image_path)
    depth_map = predict_depth(depth_model, input_image)
    depth_map_resized = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]))

    visualize_depth_map(depth_map_resized, depth_output_path)

    threshold = calculate_threshold(depth_map, percentile=60)
    foreground_mask = generate_foreground_mask(depth_map, threshold)
    refined_mask = refine_mask(foreground_mask)

    visualize_mask(refined_mask, mask_output_path)

    portrait_image = combine_foreground_background(original_image, refined_mask)

    cv2.imwrite(output_image_path, portrait_image)
    print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nPortrait -> {output_image_path}")


def process_image_folder(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpeg', '.jpg', '.png'))]

    print(f"Found {len(image_files)} images in {input_folder}. Processing...")

    for i, image_file in enumerate(image_files):
        input_image_path = os.path.join(input_folder, image_file)

        image_output_folder = os.path.join(output_folder, f"output_{i+1}")
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        depth_output_path = os.path.join(image_output_folder, f"{i+1}_depth_map.png")
        mask_output_path = os.path.join(image_output_folder, f"{i+1}_mask.png")
        portrait_output_path = os.path.join(image_output_folder, f"{i+1}_portrait.png")

        try:
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")

            apply_portrait_mode(input_image_path, portrait_output_path, depth_output_path, mask_output_path)

            print(f"Saved outputs for {image_file} to {image_output_folder}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def visualize_mask(mask, output_path="mask_visualization.png"):
    plt.imshow(mask, cmap="gray")
    plt.title("Mask Visualization")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    print(f"Mask visualization saved to {output_path}")

def visualize_depth_map(depth_map, output_path="depth_map_visualization.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap="plasma")
    plt.colorbar()
    plt.title("Depth Map Visualization")
    plt.savefig(output_path)
    plt.close()
    print(f"Depth map visualization saved to {output_path}")

process_image_folder(input_folder, output_folder)