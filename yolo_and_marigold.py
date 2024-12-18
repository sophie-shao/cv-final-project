import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt
import os

input_folder = "marigold_dataset" 
output_folder = "yolo_outputs"  
allowed_classes = ["person", "cat"]

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    print("YOLOv5 model loaded.")
    return model

def load_depth_model():
    model = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16"
    ).to("cpu")
    print("Marigold depth model loaded.")
    return model

# This is for Marigold depth map
def predict_depth(model, image):
    depth = model(image)
    depth_map = depth.prediction[0]
    return depth_map

def preprocess_image(image_path):
    image = diffusers.utils.load_image(image_path)
    return image

# This is for calculating the average depth in the YOLO-detected bounding box
def calculate_average_depth(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    cropped_region = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(cropped_region)
    print(f"Average depth in bounding box: {avg_depth:.3f}")
    return avg_depth

def generate_foreground_mask(depth_map, threshold):
    mask = (depth_map < threshold).astype(np.uint8) * 255
    return mask

def refine_mask(mask):

    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    
    return refined_mask

# This is for combining Marigold depth mask and YOLO box 
def combine_masks(mask, yolo_bbox, image_shape):

    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = yolo_bbox
    combined_mask[y1:y2, x1:x2] = 255
    combined_mask = cv2.bitwise_and(combined_mask, mask)
    combined_mask = cv2.GaussianBlur(combined_mask, (31, 31), 0)

    return combined_mask

# And then this is for combining the sharp foreground and the blurred background
def combine_sharp_and_blurred(image, mask, blur_strength=151):

    blurred_background = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    mask_3d = np.stack([mask / 255.0] * 3, axis=-1)
    sharp_foreground = image * mask_3d
    blurred_only = blurred_background * (1 - mask_3d)
    combined_image = sharp_foreground + blurred_only

    return combined_image.astype(np.uint8)

def draw_green_box(image_path, bbox, output_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(output_path, image)

    print(f"Green box image saved to {output_path}")

def apply_portrait_mode(image_path, output_image_path, depth_output_path, mask_output_path, box_output_path, depth_threshold, allowed_classes):

    yolo_model = load_yolo_model()
    depth_model = load_depth_model()

    original_image = cv2.imread(image_path)
    input_image = preprocess_image(image_path)

    depth_map = predict_depth(depth_model, input_image)
    depth_map_resized = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]))
    visualize_depth_map(depth_map_resized, depth_output_path)

    results = yolo_model(image_path)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['name'].isin(allowed_classes)]

    # If no object is detected
    if detections.empty:
        print("No allowed objects detected. Applying Marigold mask to the entire image.")
        # Apply marigold mask
        foreground_mask = generate_foreground_mask(depth_map_resized, depth_threshold)
        refined_mask = refine_mask(foreground_mask)
        visualize_mask(refined_mask, mask_output_path)
        combined_result = combine_sharp_and_blurred(original_image, refined_mask, blur_strength=151)
        cv2.imwrite(output_image_path, combined_result)
        print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nPortrait -> {output_image_path}")
        return

    accumulated_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    box_drawn = False

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        
        avg_depth = calculate_average_depth(depth_map_resized, (x1, y1, x2, y2))

        # Check if the average depth is below or equal to the threshold
        if avg_depth <= depth_threshold:
            print(f"Including object: {label} at {x1}, {y1}, {x2}, {y2} with avg depth {avg_depth:.2f} <= {depth_threshold}")
            
            foreground_mask = generate_foreground_mask(depth_map_resized, avg_depth)
            refined_mask = refine_mask(foreground_mask)
            obj_mask = combine_masks(refined_mask, (x1, y1, x2, y2), original_image.shape)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, obj_mask)

            if not box_drawn:
                draw_green_box(image_path, (x1, y1, x2, y2), box_output_path)
                box_drawn = True
        else:
            print(f"Skipping object: {label} at {x1}, {y1}, {x2}, {y2} (avg depth {avg_depth:.2f} > {depth_threshold})")

    # After everything is done:
    if np.count_nonzero(accumulated_mask) > 0:
        # We have a combined mask from one or more objects
        visualize_mask(accumulated_mask, mask_output_path)
        combined_result = combine_sharp_and_blurred(original_image, accumulated_mask, blur_strength=151)
        cv2.imwrite(output_image_path, combined_result)
        print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nPortrait -> {output_image_path}")
    else:
        # No objects met the depth threshold, so we apply Marigold mask only
        print("No allowed objects met the depth threshold. Applying Marigold mask to the entire image.")
        foreground_mask = generate_foreground_mask(depth_map_resized, depth_threshold)
        refined_mask = refine_mask(foreground_mask)
        visualize_mask(refined_mask, mask_output_path)
        combined_result = combine_sharp_and_blurred(original_image, refined_mask, blur_strength=151)
        cv2.imwrite(output_image_path, combined_result)
        print(f"Saved: Depth Map -> {depth_output_path},\nMask -> {mask_output_path},\nPortrait -> {output_image_path}")


def process_image_folder(input_folder, output_folder, allowed_classes, depth_threshold):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpeg'))]

    print(f"Found {len(image_files)} images in {input_folder}. Processing...")
    
    for i, image_file in enumerate(image_files):
        input_image_path = os.path.join(input_folder, image_file)

        image_output_folder = os.path.join(output_folder, f"output_{i+1}")
        if not os.path.exists(image_output_folder):
            os.makedirs(image_output_folder)

        depth_output_path = os.path.join(image_output_folder, f"{i+1}_depth_map.png")
        mask_output_path = os.path.join(image_output_folder, f"{i+1}_mask.png")
        portrait_output_path = os.path.join(image_output_folder, f"{i+1}_portrait.png")
        box_output_path = os.path.join(image_output_folder, f"{i+1}_box.png")

        try:
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            apply_portrait_mode(
                image_path=input_image_path,
                output_image_path=portrait_output_path,
                depth_output_path=depth_output_path,
                mask_output_path=mask_output_path,
                box_output_path=box_output_path,
                depth_threshold=depth_threshold,
                allowed_classes=allowed_classes
            )
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

process_image_folder(input_folder, output_folder, allowed_classes=allowed_classes, depth_threshold=0.5)
