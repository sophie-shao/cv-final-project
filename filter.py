import cv2
import numpy as np
from mtcnn import MTCNN

def overlay_image(background, overlay, position, size):
    """ Overlay a transparent image (PNG) onto a background at a specified position. """
    overlay_resized = cv2.resize(overlay, size)

    # Extract the alpha channel if it exists
    if overlay_resized.shape[2] == 4:
        alpha_overlay = overlay_resized[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay

        # Blend the images
        for c in range(0, 3):
            y1, y2 = position[1], position[1] + size[1]
            x1, x2 = position[0], position[0] + size[0]
            background[y1:y2, x1:x2, c] = (
                alpha_overlay * overlay_resized[:, :, c] +
                alpha_background * background[y1:y2, x1:x2, c]
            )

def add_cat_filter(image_path, ear_path, heart_path, output_path):
    detector = MTCNN()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found.")
        return
    
    # Load filter images
    ears = cv2.imread(ear_path, cv2.IMREAD_UNCHANGED)
    hearts = cv2.imread(heart_path, cv2.IMREAD_UNCHANGED)
    if ears is None or hearts is None:
        print("Error: Could not load filter images.")
        return

    # Detect faces and landmarks
    detections = detector.detect_faces(image)
    for detection in detections:
        x, y, width, height = detection['box']
        keypoints = detection['keypoints']

        # 1. Draw a green box around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # 2. Position cat ears above the eyes
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        forehead_x = (left_eye[0] + right_eye[0]) // 2
        forehead_y = min(left_eye[1], right_eye[1]) - 50  # Slightly above the eyes
        ear_size = (width, height // 2)  # Adjust ear size relative to face size
        overlay_image(image, ears, (forehead_x - ear_size[0] // 2, forehead_y - ear_size[1]), ear_size)

        # 3. Position hearts on the cheeks
        heart_size = (500, 500)
        left_cheek = (left_eye[0] - 300, left_eye[1] - 50)
        right_cheek = (right_eye[0] - 200, right_eye[1] - 50)
        overlay_image(image, hearts, left_cheek, heart_size)
        overlay_image(image, hearts, right_cheek, heart_size)

    # Save the result
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Paths
image_path = "images/IMG_8967.jpeg"  # Input image
ear_path = "images/ears.png"    # Path to cat ears PNG
heart_path = "images/hearts.png"   # Path to hearts PNG
output_path = "cat_filter_output.png"    # Output image

# Run the function
add_cat_filter(image_path, ear_path, heart_path, output_path)
