import cv2
import numpy as np
from PIL import Image
import diffusers
import torch
import matplotlib.pyplot as plt

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with actual YOLO model weights and config
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Load class labels (e.g., cat, person, food)
def load_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Detect objects with YOLO
def detect_objects(image, net, output_layers, classes):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indices

# Apply the mask and compute the average depth for detected objects
def get_subject_depth(image, boxes, depth_map):
    depths = []
    for (x, y, w, h) in boxes:
        # Extract the region corresponding to the detected object
        object_region = depth_map[y:y + h, x:x + w]
        # Compute the average depth of the object
        avg_depth = np.mean(object_region)
        depths.append(avg_depth)
    return np.mean(depths)  # Return the average depth of the subject


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



def apply_portrait_mode_with_yolo(image_path, output_path):
    # Step 1: Load the YOLO model
    net, output_layers = load_yolo_model()
    classes = load_classes()

    # Step 2: Load the depth model (Marigold)
    depth_model = load_depth_model()

    # Step 3: Preprocess the image and perform object detection
    original_image = cv2.imread(image_path)
    input_image = preprocess_image(image_path)
    depth_map = predict_depth(depth_model, input_image)

    # Step 4: Detect objects (person, cat, food, etc.) with YOLO
    boxes, confidences, class_ids, indices = detect_objects(original_image, net, output_layers, classes)

    # Step 5: Get the average depth of the subject (person, cat, etc.)
    subject_depth = get_subject_depth(original_image, boxes, depth_map)
    print(f"Average subject depth: {subject_depth}")

    # Step 6: Create a refined foreground mask based on dynamic thresholding
    threshold = subject_depth  # Dynamic threshold based on subject's depth
    foreground_mask = generate_foreground_mask(depth_map, threshold)
    refined_mask = refine_mask(foreground_mask)

    # Step 7: Combine the sharp foreground and blurred background
    portrait_image = combine_foreground_background(original_image, refined_mask)

    # Step 8: Save the resulting image
    cv2.imwrite(output_path, portrait_image)
    print(f"Portrait mode image saved to {output_path}")

# Example usage
input_image_path = "images/your_image.jpeg"  # Replace with your image path
output_image_path = "portrait_mode_output.jpg"

apply_portrait_mode_with_yolo(input_image_path, output_image_path)
