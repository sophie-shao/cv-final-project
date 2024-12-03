# cv-final-project

Objective: Create a filter that sharpens the foreground and blurs the background (basically the portrait filter on iphone)

  Pre-trained model choice: Openface/Facenet

    Code outline: 
      1. upload pre-trained model
      
        import dlib
        import numpy as np
        import cv2
        
      2. preprocess the image so that we separate the foreground and background from each other and apply filters

        cv2.imread(image)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
          1. detect landmarks on the face
              landmarks = predictor(gray, face)
              points = [(p.x, p.y) for p in landmarks.parts()]
          2. create face mask
            face_mask = get_face_mask(image, points)
          3. sharpen the foreground and blur the background
            ex: sharpening_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                blurring_kernel = GaussianBlur(image, (21,21), 0)
          5. combine

  Pre-trained model choice: MTCNN

    Code outline:

  Pre-trained model choice: VGG-Face
    Restricted for commerical use :(

    Code outline:
