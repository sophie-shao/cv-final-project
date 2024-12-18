# cv-final-project

Objective: Create a filter that sharpens the foreground and blurs the background (basically the portrait filter on iphone)

The problem we set out to solve is to create an effect portrait mode filter that can identify the subject in the foreground (e.g., a person, a cat, or an object) and blur the background, imitating the shallow depth of field photography achieved by professional cameras. We sought to achieve this effect in two ways, through a traditional approach and through machine learning, and in comparing our results we aim to generate a better understanding of the relative merits of each approach and the factors that influence performance in each case.

(1) We used stereo vision to estimate depth from two different view of a scene and generate a disparity map. This provides depth information that can help distinguish between the foreground and background.

(2) We used pre-trained Marigold diffusers to estimate depth in a single image and combined it with YOLO object detection. This combination allowed us to detect the main subject, calculate the average depth of the detected object, and then extract the foregound by thresholding values based on the average depth in the vicinity of the subject.

